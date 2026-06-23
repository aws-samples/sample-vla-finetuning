/**
 * PatternBStack — IL axis, Pattern B (SageMaker Training Job).
 *
 * Pattern B runs the generalized `vla-ft` container as a *managed* SageMaker
 * Training Job: SageMaker provides the GPU host, network, dataset download,
 * checkpoint↔S3 sync (Managed-Spot auto-resume), and model-artifact upload. So the
 * platform's job here is **identity + wiring**, not infrastructure — there is no
 * VPC/EC2/AzSelector at this layer (those belong to Pattern A Batch and the RL
 * axis, which place capacity in our own VPC).
 *
 * What this stack creates:
 *   - a SageMaker **execution role** (trusted by sagemaker.amazonaws.com) that the
 *     training job assumes. It attaches the SharedBaseStack `jobBasePolicy`
 *     (read dataBucket / read-write artifactBucket / pull both ECR repos) and
 *     layers the SageMaker-specific permissions a BYOC training job needs on top:
 *     CloudWatch Logs + metric publication.
 *
 * Why least-privilege instead of AmazonSageMakerFullAccess: the verified smoke role
 * used FullAccess, but a BYOC training job that only reads a dataset from S3, pulls
 * one image from ECR, writes checkpoints/model back to S3, and emits logs/metrics
 * needs exactly the set below. The base stack comment states the intent — pattern
 * stacks attach jobBasePolicy then add service-specific permissions — so this is the
 * designed seam, and it keeps the reference architecture honest about scope.
 * (the *container* lock is untouched; this is the
 * IAM envelope around it, which the verified path over-granted.)
 *
 * The HF token is NOT granted here: launch.py reads it (SSM/Secrets) on the
 * submitting side and injects it as a job environment variable, exactly as the
 * verified run did — the training job itself never calls SSM.
 *
 * Synthesizes with no credentials. Cost at this layer is ~0 (an IAM role); the GPU
 * cost is incurred only when a training job actually runs.
 */
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';
import { SharedBaseStack } from '../shared/base-stack';
import { TrainingNotifications } from '../shared/notifications';

export interface PatternBStackProps extends cdk.StackProps {
  /** The shared base stack whose buckets, ECR repos, and jobBasePolicy this imports. */
  readonly base: SharedBaseStack;
  /**
   * Prefix for physical resource names (must match the base stack's, so a multi-
   * deployment account stays isolated). Default 'pai'.
   */
  readonly namePrefix?: string;
  /**
   * Optional stable name for the execution role. Default: CDK-generated (avoids
   * cross-region IAM name collisions, since IAM is global). Set this only when an
   * operator wants a predictable `--role` ARN for launch.py.
   */
  readonly executionRoleName?: string;
  /**
   * Extra S3 bucket ARNs the execution role may READ datasets from, beyond the
   * platform dataBucket. Use when pointing a run at a pre-existing LeRobot dataset
   * that has not been copied into the platform bucket (e.g. the verified
   * openarm-lift dataset bucket). Pass the bucket ARN; object-level access is added
   * automatically. Read-only — never write.
   */
  readonly extraDatasetReadArns?: string[];
  /**
   * Email address to notify when a training job reaches a terminal state
   * (Completed / Failed / Stopped). When set, an SNS topic + EventBridge rule are
   * created and the address is subscribed (one-time confirmation click required).
   * When omitted, no notification resources are created. Wire via
   * `-c notifyEmail=you@example.com`.
   */
  readonly notifyEmail?: string;
  /**
   * Job-name prefix the notification rule filters on. The account is SMUS-shared,
   * so this keeps alerts to this platform's jobs. Default 'vla-ft-' (matches
   * launch.py's `vla-ft-<policy>-<ts>` naming).
   */
  readonly notifyJobNamePrefix?: string;
}

export class PatternBStack extends cdk.Stack {
  /** SageMaker execution role ARN — pass to launch.py `--role`. */
  public readonly executionRole: iam.Role;
  /** Job-completion notifications (SNS topic + EventBridge rule); set only when notifyEmail is given. */
  public readonly notifications?: TrainingNotifications;

  constructor(scope: Construct, id: string, props: PatternBStackProps) {
    super(scope, id, props);

    const { base } = props;
    const namePrefix = props.namePrefix ?? 'pai';

    this.executionRole = new iam.Role(this, 'ExecutionRole', {
      roleName: props.executionRoleName,
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      description:
        'PAI Training Platform - Pattern B (SageMaker Training Job) execution role for the vla-ft IL container',
      // Base envelope: read dataBucket, read/write artifactBucket, pull ECR repos.
      managedPolicies: [base.jobBasePolicy],
    });

    // CloudWatch Logs: SageMaker training jobs stream to /aws/sagemaker/*.
    this.executionRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'TrainingJobLogs',
        actions: [
          'logs:CreateLogGroup',
          'logs:CreateLogStream',
          'logs:PutLogEvents',
          'logs:DescribeLogStreams',
        ],
        resources: [
          `arn:${this.partition}:logs:${this.region}:${this.account}:log-group:/aws/sagemaker/*`,
        ],
      }),
    );

    // CloudWatch metrics: SageMaker publishes training metrics. PutMetricData has no
    // resource-level scoping, so constrain by namespace condition to the SageMaker
    // namespaces (write-only, low-risk).
    this.executionRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'TrainingJobMetrics',
        actions: ['cloudwatch:PutMetricData'],
        resources: ['*'],
        conditions: {
          StringEquals: {
            'cloudwatch:namespace': ['/aws/sagemaker/TrainingJobs', 'aws/sagemaker/TrainingJobs'],
          },
        },
      }),
    );

    // Optional read on pre-existing dataset buckets (e.g. the verified openarm-lift
    // dataset not yet copied into the platform dataBucket).
    if (props.extraDatasetReadArns && props.extraDatasetReadArns.length > 0) {
      this.executionRole.addToPolicy(
        new iam.PolicyStatement({
          sid: 'ExtraDatasetRead',
          actions: ['s3:GetObject', 's3:ListBucket', 's3:GetBucketLocation'],
          resources: props.extraDatasetReadArns.flatMap((arn) => [arn, `${arn}/*`]),
        }),
      );
    }

    // --- Job-completion notifications (opt-in via notifyEmail) ---
    // A SageMaker Training Job is not a CFN resource, so its completion can't be
    // observed by this stack directly. EventBridge emits a state-change event we
    // route to SNS → email. Created only when an address is supplied.
    if (base.notificationTopic) {
      this.notifications = new TrainingNotifications(this, 'Notifications', {
        topic: base.notificationTopic,
      });
      this.notifications.addSageMakerTrainingJobRule(props.notifyJobNamePrefix ?? 'vla-ft-');
    }

    // --- Outputs: everything launch.py needs, in one place ---
    new cdk.CfnOutput(this, 'ExecutionRoleArn', {
      value: this.executionRole.roleArn,
      description: 'Pass to launch.py --role',
    });
    new cdk.CfnOutput(this, 'ImageUriHint', {
      value: `${base.vlaFtRepo.repositoryUri}:latest`,
      description: 'Build target for containers/vla-ft/build.sh; pass to launch.py --image-uri',
    });
    new cdk.CfnOutput(this, 'OutputS3Hint', {
      value: `s3://${base.artifactBucket.bucketName}/vla-ft`,
      description: 'Pass to launch.py --output-s3 (checkpoints + model.tar.gz land here)',
    });
    new cdk.CfnOutput(this, 'DataBucketName', {
      value: base.dataBucket.bucketName,
      description: 'Upload LeRobot v3 datasets here; reference as launch.py --dataset-s3',
    });
  }
}
