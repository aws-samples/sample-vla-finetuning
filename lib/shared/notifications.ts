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
 * source now; a `addBatchJobRule()` sibling can route Pattern A / RL Batch jobs to
 * the same topic later without a second email subscription.
 *
 * Cost is ~0: SNS + EventBridge are per-message priced and a few job completions a
 * day is negligible. The email subscription requires a one-time confirmation click
 * (AWS sends a "Subscription Confirmation" email on first deploy).
 */
import * as cdk from 'aws-cdk-lib';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as subs from 'aws-cdk-lib/aws-sns-subscriptions';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import { Construct } from 'constructs';

export interface TrainingNotificationsProps {
  /** Prefix for the topic's physical name. Default 'pai'. */
  readonly namePrefix?: string;
  /**
   * Email address to subscribe to the topic. If omitted, the topic is still
   * created (so rules can attach) but no subscription is made — useful when a
   * caller wires the topic into other targets, or defers the email to a later
   * deploy via `-c notifyEmail=...`.
   */
  readonly notifyEmail?: string;
}

export class TrainingNotifications extends Construct {
  /** Topic that all training-job terminal-state events publish to. */
  public readonly topic: sns.Topic;

  constructor(scope: Construct, id: string, props: TrainingNotificationsProps = {}) {
    super(scope, id);

    const prefix = props.namePrefix ?? 'pai';

    this.topic = new sns.Topic(this, 'Topic', {
      topicName: `${prefix}-training-notifications`,
      displayName: 'PAI Training Platform',
    });

    if (props.notifyEmail) {
      this.topic.addSubscription(new subs.EmailSubscription(props.notifyEmail));
    }
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
    // event (e.g. FailureReason on a success) render as empty strings.
    const message = events.RuleTargetInput.fromText(
      [
        `PAI Training Platform — SageMaker training job ${events.EventField.fromPath('$.detail.TrainingJobStatus')}`,
        '',
        `Job:       ${events.EventField.fromPath('$.detail.TrainingJobName')}`,
        `Status:    ${events.EventField.fromPath('$.detail.TrainingJobStatus')}`,
        `Region:    ${events.EventField.fromPath('$.region')}`,
        `Time:      ${events.EventField.fromPath('$.time')}`,
        `Artifacts: ${events.EventField.fromPath('$.detail.ModelArtifacts.S3ModelArtifacts')}`,
        `Failure:   ${events.EventField.fromPath('$.detail.FailureReason')}`,
      ].join('\n'),
    );

    rule.addTarget(new targets.SnsTopic(this.topic, { message }));
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
  addBatchJobRule(jobQueueArn: string, jobNamePrefix: string): events.Rule {
    const rule = new events.Rule(this, 'BatchJobRule', {
      description:
        `Route Batch job terminal states (queue, name prefix "${jobNamePrefix}") to ${this.topic.topicName}`,
      eventPattern: {
        source: ['aws.batch'],
        detailType: ['Batch Job State Change'],
        detail: {
          jobqueue: [jobQueueArn],
          jobName: events.Match.prefix(jobNamePrefix),
          status: ['FAILED', 'SUCCEEDED'],
        },
      },
    });

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

    rule.addTarget(new targets.SnsTopic(this.topic, { message }));
    return rule;
  }
}
