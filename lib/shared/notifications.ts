/**
 * TrainingNotifications — SNS-backed completion/failure alerts for training jobs.
 *
 * The platform runs training jobs on services that are NOT CloudFormation
 * resources (SageMaker Training Jobs for Pattern B; Batch jobs for Pattern A / RL).
 * A job's lifecycle therefore can't be observed by a stack the way a CFN resource
 * can. The portable way to react to "job finished" across all of them is
 * EventBridge: each service emits a *State Change* event, and a rule routes the
 * terminal states (Completed / Failed / Stopped) to one SNS topic. An email
 * subscription on that topic gives the operator a human notification. This is the
 * concrete implementation of the architecture's orchestrator "notify" step, pulled
 * forward so Pattern B operators get alerts on long (~5h) fine-tunes today —
 * managed EventBridge + SNS, no Lambda/poller.
 *
 * Reusable across patterns: `addSageMakerTrainingJobRule()` wires the SageMaker
 * source; `addBatchJobRule()` routes Pattern A / RL Batch jobs to the same topic.
 *
 * ★ Single topic owner: the SNS topic is created ONCE in the Base stack and passed in
 * here (props.topic). An earlier version created a fixed-name topic per stack, which
 * collided (AWS::SNS::Topic AlreadyExists) as soon as a second stack also got notifyEmail
 * — e.g. Pattern B (SageMaker) and Pattern A (Batch) both notifying. Importing the shared
 * topic keeps one subscription and lets every pattern add only its own EventBridge rule.
 *
 * Cost is ~0: SNS + EventBridge are per-message priced and a few job completions a
 * day is negligible. The email subscription requires a one-time confirmation click
 * (AWS sends a "Subscription Confirmation" email on first deploy).
 */
import * as cdk from 'aws-cdk-lib';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import { Construct } from 'constructs';

export interface TrainingNotificationsProps {
  /**
   * The shared SNS topic to route alerts to — created once in the Base stack and
   * imported by each pattern stack (base.notificationTopic). Owning the topic in one
   * place is what prevents the fixed-name AlreadyExists collision across stacks.
   */
  readonly topic: sns.ITopic;
}

export class TrainingNotifications extends Construct {
  /** Topic that all training-job terminal-state events publish to (owned by Base). */
  public readonly topic: sns.ITopic;

  /**
   * Dead-letter queue for events EventBridge fails to deliver to the SNS target.
   *
   * Why this exists: a rule whose target invocation fails (e.g. an InputTransformer
   * that produced an unusable payload, or a transient SNS/permission error) increments
   * the rule's FailedInvocations metric and otherwise *silently drops the event* — there
   * is no record of which event failed or why. Attaching a DLQ captures the original
   * event plus error metadata (ERROR_CODE / ERROR_MESSAGE / RULE_ARN / TARGET_ARN in the
   * message attributes), so a missed alert can be diagnosed from concrete data instead of
   * inferred from metrics. One queue per stack's construct (a construct only ever adds one
   * of the SageMaker/Batch rules), shared by whatever rule that stack registers.
   */
  public readonly deadLetterQueue: sqs.Queue;

  constructor(scope: Construct, id: string, props: TrainingNotificationsProps) {
    super(scope, id);
    this.topic = props.topic;

    this.deadLetterQueue = new sqs.Queue(this, 'NotificationDlq', {
      retentionPeriod: cdk.Duration.days(14), // max — keep failed-delivery evidence long enough to investigate
      removalPolicy: cdk.RemovalPolicy.DESTROY, // diagnostic only; nothing to preserve on teardown
    });
  }

  /**
   * Route SageMaker Training Job terminal-state changes whose name starts with
   * `jobNamePrefix` to this topic, with a human-readable email body. The account
   * is SMUS-shared, so the name-prefix filter keeps alerts to this platform's jobs
   * (launch.py names them `vla-ft-<policy>-<ts>`).
   */
  addSageMakerTrainingJobRule(jobNamePrefix: string): events.Rule {
    const rule = new events.Rule(this, 'SageMakerTrainingJobRule', {
      description:
        `Route SageMaker Training Job terminal states (name prefix "${jobNamePrefix}") to ${this.topic.topicName}`,
      eventPattern: {
        source: ['aws.sagemaker'],
        detailType: ['SageMaker Training Job State Change'],
        detail: {
          TrainingJobName: events.Match.prefix(jobNamePrefix),
          TrainingJobStatus: ['Completed', 'Failed', 'Stopped'],
        },
      },
    });

    // Transform the raw event into a readable email. Fields absent on a given
    // event (e.g. ModelArtifacts on a failure) render as empty strings.
    //
    // ★ FailureReason is deliberately NOT interpolated. RuleTargetInput.fromText emits an
    // EventBridge InputTransformer whose InputTemplate is a JSON-quoted string; SageMaker's
    // FailureReason on an OOM is a long multi-line traceback with raw newlines, double
    // quotes and backslashes. Substituting it produces an InputTemplate that is no longer
    // valid JSON, so EventBridge drops the target invocation entirely (FailedInvocations++,
    // SNS NumberOfMessagesPublished stays 0 — no email at all, not even a truncated one).
    // We surface a console deep-link instead: every interpolated field below (job name,
    // region) is special-char-free, so the template stays valid, and the operator clicks
    // through to the full failure reason in the SageMaker console.
    const message = events.RuleTargetInput.fromText(
      [
        `PAI Training Platform — SageMaker training job ${events.EventField.fromPath('$.detail.TrainingJobStatus')}`,
        '',
        `Job:       ${events.EventField.fromPath('$.detail.TrainingJobName')}`,
        `Status:    ${events.EventField.fromPath('$.detail.TrainingJobStatus')}`,
        `Region:    ${events.EventField.fromPath('$.region')}`,
        `Time:      ${events.EventField.fromPath('$.time')}`,
        `Artifacts: ${events.EventField.fromPath('$.detail.ModelArtifacts.S3ModelArtifacts')}`,
        '',
        // Failure reason (full traceback) is on the console page — see "Failure reason".
        `Details:   https://${events.EventField.fromPath('$.region')}.console.aws.amazon.com/sagemaker/home?region=${events.EventField.fromPath('$.region')}#/jobs/${events.EventField.fromPath('$.detail.TrainingJobName')}`,
      ].join('\n'),
    );

    rule.addTarget(new targets.SnsTopic(this.topic, { message, deadLetterQueue: this.deadLetterQueue }));
    return rule;
  }

  /**
   * Route AWS Batch job terminal-state changes (FAILED / SUCCEEDED) on the given
   * queue, whose name starts with `jobNamePrefix`, to this topic with a readable
   * email body. The Batch counterpart of addSageMakerTrainingJobRule — Pattern A
   * (and RL Batch) use this to reach the same topic without a second subscription.
   *
   * Batch emits 'Batch Job State Change' on every transition; we filter to the two
   * terminal states. The queue-ARN filter keeps alerts to this stack's queue (the
   * account is SMUS-shared), and the name-prefix mirrors the SageMaker rule.
   */
  addBatchJobRule(jobQueueArn: string | string[], jobNamePrefix: string): events.Rule {
    // Accept one queue ARN or several (a stack may own a Spot + On-Demand queue pair, and a
    // job lands on whichever was selected at SubmitJob). The EventBridge `jobqueue` filter
    // is a match-any list, so one rule covers all of them.
    const jobqueue = Array.isArray(jobQueueArn) ? jobQueueArn : [jobQueueArn];
    const rule = new events.Rule(this, 'BatchJobRule', {
      description:
        `Route Batch job terminal states (queue, name prefix "${jobNamePrefix}") to ${this.topic.topicName}`,
      eventPattern: {
        source: ['aws.batch'],
        detailType: ['Batch Job State Change'],
        detail: {
          jobqueue,
          jobName: events.Match.prefix(jobNamePrefix),
          status: ['FAILED', 'SUCCEEDED'],
        },
      },
    });

    // Batch's statusReason is a short single-line string (e.g. "Essential container in
    // task exited") — unlike SageMaker's FailureReason it does not carry a multi-line
    // traceback, so interpolating it does not threaten InputTemplate validity. The DLQ is
    // attached anyway so any future delivery failure is diagnosable, not silently dropped.
    const message = events.RuleTargetInput.fromText(
      [
        `PAI Training Platform — Batch job ${events.EventField.fromPath('$.detail.status')}`,
        '',
        `Job:       ${events.EventField.fromPath('$.detail.jobName')}`,
        `Status:    ${events.EventField.fromPath('$.detail.status')}`,
        `Reason:    ${events.EventField.fromPath('$.detail.statusReason')}`,
        `Region:    ${events.EventField.fromPath('$.region')}`,
        `Time:      ${events.EventField.fromPath('$.time')}`,
      ].join('\n'),
    );

    rule.addTarget(new targets.SnsTopic(this.topic, { message, deadLetterQueue: this.deadLetterQueue }));
    return rule;
  }
}
