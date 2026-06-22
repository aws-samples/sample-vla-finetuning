/**
 * OrchestratorStack — Phase 4, the deterministic Step Functions orchestrator.
 *
 * This is the **scale-gated graduation** of the launcher's decision logic
 * (ARCHITECTURE §3). The launcher (`vla_ft_cli.py`, Phase 6) already turns one intent
 * into a backend choice as a smart default; this stack promotes that *same* logic into
 * a durable, auditable Step Functions state machine for when the workload is
 * multi-job / multi-user and wants retries + execution history + async fan-out. It is
 * intentionally NOT in `bin/app.ts` by default (deploy-gated, like the HyperPod
 * stacks) — the test IS the synth gate.
 *
 * Determinism (the design constraint): the state machine has **no LLM / no Bedrock** in
 * its path. Two Lambda steps, both pure functions of their input:
 *   1. Plan  (`orchestrator_plan.handler`) — classify → profile → decide. It imports
 *      `vla_ft_decide` VERBATIM — the SAME rule-table module the launcher calls — so the
 *      decision logic exists in exactly one place.
 *   2. Submit (`orchestrator_submit.handler`) — dispatches the chosen backend, faithful
 *      to the verified launchers (Batch submitted directly; SageMaker handed off as the
 *      exact launch.py command rather than forking the SDK estimator).
 * A Choice state routes on the plan's deterministic `runnable` flag: Pattern A/B run,
 * Pattern C (code+synth only) falls through to a Recommend pass-state. A final SNS
 * publish notifies on the outcome (reusing the platform's notification topic pattern).
 *
 *   Start → Plan → Choice( runnable? )
 *                    ├─ true  → Submit → Notify(Succeed)
 *                    └─ false → Recommend(Pattern C) → Notify(Succeed)
 *                  Plan/Submit error → Notify(Fail)
 *
 * Wiring is injected into the Submit Lambda as ENVIRONMENT VARIABLES at synth from the
 * imported pattern stacks (queue/jobdef/role/image/output ARNs, HF-token SSM name), so
 * the Lambda is deterministic and needs no CloudFormation read IAM at runtime. Both
 * Lambdas share one asset = `containers/vla-ft/` (the SAME train.py the container ships,
 * so the source_dir equivalent never drifts).
 *
 * Cost: ~0 at synth. When deployed, Step Functions + Lambda + SNS are per-execution
 * priced; the GPU cost is incurred only by a job the Submit step actually launches.
 */
import * as cdk from 'aws-cdk-lib';
import * as path from 'path';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import { Construct } from 'constructs';
import { SharedBaseStack } from '../shared/base-stack';
import { PatternAStack } from '../il/pattern-a-stack';
import { PatternBStack } from '../il/pattern-b-stack';
import { GrootPatternAStack } from '../il/gr00t-pattern-a-stack';
import { RlPatternAStack } from '../rl/pattern-a-stack';
import { TrainingNotifications } from '../shared/notifications';

export interface OrchestratorStackProps extends cdk.StackProps {
  /** Shared base (buckets, ECR, jobBasePolicy) — for the Lambda's S3/ECR access envelope. */
  readonly base: SharedBaseStack;
  /** IL Pattern A (Batch) stack — its queue/jobdef/code/output ARNs are injected into Submit. */
  readonly ilPatternA: PatternAStack;
  /** IL Pattern B (SageMaker) stack — its role/image/output are injected for the handoff command. */
  readonly ilPatternB: PatternBStack;
  /** RL Pattern A (Batch) stack — its queue/jobdef/output ARNs are injected into Submit. */
  readonly rlPatternA: RlPatternAStack;
  /** GR00T N1.7 Pattern A (Batch) stack — its queue/jobdef/output ARNs are injected into Submit. */
  readonly grootPatternA: GrootPatternAStack;
  /** Prefix for physical resource names (must match the base stack's). Default 'pai'. */
  readonly namePrefix?: string;
  /** Email to notify on orchestrator-run outcome (opt-in; reuses TrainingNotifications). */
  readonly notifyEmail?: string;
  /** SSM SecureString name holding the gated-backbone HF token. Default '/pai/hf-token'. */
  readonly hfTokenSsm?: string;
  /** Region of the HF-token SSM parameter. Default 'us-east-1' (where the verified run reads it). */
  readonly hfTokenSsmRegion?: string;
}

export class OrchestratorStack extends cdk.Stack {
  public readonly stateMachine: sfn.StateMachine;
  public readonly planFn: lambda.Function;
  public readonly submitFn: lambda.Function;
  public readonly notifications?: TrainingNotifications;

  constructor(scope: Construct, id: string, props: OrchestratorStackProps) {
    super(scope, id, props);

    const { base, ilPatternA, ilPatternB, rlPatternA, grootPatternA } = props;
    const namePrefix = props.namePrefix ?? 'pai';
    const hfTokenSsm = props.hfTokenSsm ?? '/pai/hf-token';
    const hfTokenSsmRegion = props.hfTokenSsmRegion ?? 'us-east-1';

    // Both Lambdas bundle the containers/vla-ft Python (the SAME files the container
    // ships — train.py is not copied, so it cannot drift from the verified lock). Exclude
    // the heavy/irrelevant bits so the asset stays small.
    const code = lambda.Code.fromAsset(path.join(__dirname, '..', '..', 'containers', 'vla-ft'), {
      exclude: ['docker', '__pycache__', '*.pyc', 'build.sh', 'README.md', 'test_*.py'],
    });

    // --- 1. Plan Lambda: classify → profile → decide (pure; no AWS calls) ---
    this.planFn = new lambda.Function(this, 'PlanFn', {
      functionName: `${namePrefix}-orchestrator-plan`,
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'orchestrator_plan.handler',
      code,
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      description:
        'PAI orchestrator plan step: classify IL/RL, profile, decide backend (imports vla_ft_decide verbatim)',
    });

    // --- 2. Submit Lambda: dispatch the chosen backend, faithful to the launchers ---
    this.submitFn = new lambda.Function(this, 'SubmitFn', {
      functionName: `${namePrefix}-orchestrator-submit`,
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'orchestrator_submit.handler',
      code,
      timeout: cdk.Duration.minutes(2),
      memorySize: 512,
      description:
        'PAI orchestrator submit step: submit Batch (IL-A/RL-A) faithfully, hand off SageMaker (B) as launch.py',
      environment: {
        REGION: this.region,
        // IL Pattern A (Batch) wiring — the four batch_launch.py needs.
        IL_A_JOB_QUEUE: ilPatternA.jobQueue.jobQueueArn,
        IL_A_JOB_DEFINITION: ilPatternA.jobDefinition.jobDefinitionArn,
        IL_A_CODE_S3: `s3://${base.artifactBucket.bucketName}/vla-ft-code/train.py`,
        IL_A_OUTPUT_S3: `s3://${base.artifactBucket.bucketName}/vla-ft`,
        // IL Pattern B (SageMaker) wiring — for the handoff command (not a direct submit).
        B_EXECUTION_ROLE: ilPatternB.executionRole.roleArn,
        B_IMAGE_URI: `${base.vlaFtRepo.repositoryUri}:latest`,
        B_OUTPUT_S3: `s3://${base.artifactBucket.bucketName}/vla-ft`,
        // RL Pattern A (Batch) wiring — the three rl_launch.py needs.
        RL_A_JOB_QUEUE: rlPatternA.jobQueue.jobQueueArn,
        RL_A_JOB_DEFINITION: rlPatternA.jobDefinition.jobDefinitionArn,
        RL_A_OUTPUT_S3: `s3://${base.artifactBucket.bucketName}/isaac-rl`,
        // GR00T N1.7 Pattern A (Batch) wiring — the three gr00t_launch.py needs (no code
        // upload: the GR00T trainer is baked in the image).
        GROOT_A_JOB_QUEUE: grootPatternA.jobQueue.jobQueueArn,
        GROOT_A_JOB_DEFINITION: grootPatternA.jobDefinition.jobDefinitionArn,
        GROOT_A_OUTPUT_S3: `s3://${base.artifactBucket.bucketName}/gr00t-n17`,
        // HF token (read by the Lambda, injected as job env — never into SFN history).
        HF_TOKEN_SSM: hfTokenSsm,
        HF_TOKEN_SSM_REGION: hfTokenSsmRegion,
      },
    });

    // Submit Lambda IAM — least-privilege, scoped to exactly what the launchers do:
    //   - submit Batch jobs on the two platform queues,
    //   - upload the verified train.py to the artifact bucket (the source_dir equivalent),
    //   - read the gated-backbone HF token from SSM.
    this.submitFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: 'SubmitBatchJobs',
        actions: ['batch:SubmitJob'],
        resources: [
          ilPatternA.jobQueue.jobQueueArn,
          ilPatternA.jobDefinition.jobDefinitionArn,
          rlPatternA.jobQueue.jobQueueArn,
          rlPatternA.jobDefinition.jobDefinitionArn,
          grootPatternA.jobQueue.jobQueueArn,
          grootPatternA.jobDefinition.jobDefinitionArn,
        ],
      }),
    );
    base.artifactBucket.grantPut(this.submitFn, 'vla-ft-code/*');
    this.submitFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: 'ReadHfToken',
        actions: ['ssm:GetParameter'],
        resources: [
          `arn:${this.partition}:ssm:${hfTokenSsmRegion}:${this.account}:parameter${
            hfTokenSsm.startsWith('/') ? '' : '/'
          }${hfTokenSsm}`,
        ],
      }),
    );

    // --- Notifications topic (reused construct; opt-in email) ---
    this.notifications = new TrainingNotifications(this, 'Notifications', {
      namePrefix,
      notifyEmail: props.notifyEmail,
    });

    // --- State machine definition ---
    const plan = new tasks.LambdaInvoke(this, 'Plan', {
      lambdaFunction: this.planFn,
      // Replace the state I/O with the Lambda's return payload (the plan envelope).
      outputPath: '$.Payload',
    });

    const submit = new tasks.LambdaInvoke(this, 'Submit', {
      lambdaFunction: this.submitFn,
      // The Submit Lambda reads the whole plan envelope; keep its result under $.result.
      payloadResponseOnly: true,
      resultPath: '$.result',
    });

    const recommend = new sfn.Pass(this, 'RecommendOnly', {
      comment:
        'Pattern C (HyperPod) is code+synth only - not a runnable backend. The plan is a recommendation.',
      result: sfn.Result.fromObject({
        status: 'recommendation',
        reason:
          'Chosen backend is Pattern C (HyperPod), which is code+synth only (not in bin/app.ts). ' +
          'Reduce footprint (expert-only / LoRA / smaller model / single node) for a runnable A/B path.',
      }),
      resultPath: '$.result',
    });

    const notifySuccess = new tasks.SnsPublish(this, 'NotifyOutcome', {
      topic: this.notifications.topic,
      subject: 'PAI orchestrator - run planned',
      message: sfn.TaskInput.fromJsonPathAt('$'),
    });

    const notifyFailure = new tasks.SnsPublish(this, 'NotifyFailure', {
      topic: this.notifications.topic,
      subject: 'PAI orchestrator - run FAILED',
      message: sfn.TaskInput.fromJsonPathAt('$'),
    });
    const fail = new sfn.Fail(this, 'Fail', {
      cause: 'Orchestrator plan or submit step failed (see the failure notification).',
      error: 'OrchestratorError',
    });
    notifyFailure.next(fail);

    // Plan/Submit errors route to the failure notification (durable retries are the
    // reason this phase exists at scale — Lambda invoke has SFN's built-in retry).
    plan.addCatch(notifyFailure, { resultPath: '$.error' });
    submit.addCatch(notifyFailure, { resultPath: '$.error' });

    const runnableChoice = new sfn.Choice(this, 'Runnable?')
      .when(sfn.Condition.booleanEquals('$.runnable', true), submit.next(notifySuccess))
      .otherwise(recommend.next(notifySuccess));

    const definition = plan.next(runnableChoice);

    this.stateMachine = new sfn.StateMachine(this, 'StateMachine', {
      stateMachineName: `${namePrefix}-vla-ft-orchestrator`,
      definitionBody: sfn.DefinitionBody.fromChainable(definition),
      timeout: cdk.Duration.minutes(10),
      comment:
        'Deterministic IL+RL intent to backend auto-select to submit. No LLM in path; ' +
        'decision logic = vla_ft_decide (single source, shared with the launcher).',
    });

    // --- Outputs ---
    new cdk.CfnOutput(this, 'StateMachineArn', {
      value: this.stateMachine.stateMachineArn,
      description: 'Start an execution with a vla_ft_cli-style intent JSON as input',
    });
    new cdk.CfnOutput(this, 'NotificationTopicArn', {
      value: this.notifications.topic.topicArn,
      description: 'SNS topic for orchestrator-run outcome notifications',
    });
  }
}
