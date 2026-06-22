/**
 * RlPatternAStack — RL axis, Pattern A (AWS Batch + g6e Spot).
 *
 * The RL counterpart of the IL PatternAStack, and the build that closes Phase 3's
 * Coverage gap ("IL & RL, both real"). It runs NVIDIA Isaac Lab **headless PPO** (the
 * `isaac-lab-rl` container) on a single-GPU Batch Spot node — the rule table's RL
 * entry for "single-GPU env count, short" (ARCHITECTURE §3.1). Like IL Pattern A, this
 * stack OWNS the Compute Environment / Job Queue / Job Definition — exactly the CE/JQ/JD
 * the source asset (a verified Isaac Lab infra template) left as manual console steps
 * while automating only the IAM / launch-template / SG. That manual gap is what this fills.
 *
 *
 * What's different from IL Pattern A (intentionally kept minimal — same skeleton,
 * lower review risk):
 *  - Image is the RL container (`base.isaacLabRlRepo`), not the vla-ft image.
 *  - The job command is `rl_train_bootstrap.py` (injected via `python3 -c` at synth,
 *    so the image is never rebuilt to carry orchestration glue).
 *  - The "intent" is a TASK, not a dataset — there is no dataset to download, so the
 *    container override env is the `RL_*` contract (task / experiment / output / GPUs),
 *    and there is no extra-dataset-read IAM. The jobBasePolicy's artifact read/write is
 *    all the RL job needs (it only writes policies/ONNX; the simulator makes its own data).
 *  - Longer default timeout (RL PPO runs longer than a small IL fine-tune).
 *
 * Capacity / resume strategy is identical to IL Pattern A: a managed Spot CE with
 * SPOT_PRICE_CAPACITY_OPTIMIZED across a g6e→g5 single-GPU fallback is Batch's native
 * answer to insufficient-capacity, so no AzSelector. The rsl_rl `logs/` dir is symlinked
 * onto the shared EFS by the bootstrap, so a Spot reclaim + Batch retry resumes PPO from
 * the last checkpoint (`--resume`). Single-node only: multi-node distributed PPO is the
 * RL HyperPod stack (Pattern C). Synthesizes with no credentials; the CE scales to 0
 * vCPUs when idle, so GPU cost is incurred only while a job runs.
 */
import * as cdk from 'aws-cdk-lib';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as batch from 'aws-cdk-lib/aws-batch';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import { Construct } from 'constructs';
import { SharedBaseStack } from '../shared/base-stack';
import { TrainingNotifications } from '../shared/notifications';

export interface RlPatternAStackProps extends cdk.StackProps {
  /** The shared base stack whose VPC, EFS, ECR repo, buckets, and jobBasePolicy this imports. */
  readonly base: SharedBaseStack;
  /** Prefix for physical resource names (must match the base stack's). Default 'pai'. */
  readonly namePrefix?: string;
  /**
   * Single-GPU instance fallback for the Compute Environment, highest priority first.
   * Pattern A is the single-GPU tier; the reference task (H1 rough, a small MLP on
   * proprioception) fits one L40S comfortably. Spot SPOT_PRICE_CAPACITY_OPTIMIZED
   * picks the cheapest available. For a multi-GPU run (many parallel envs), deploy
   * with a 4-GPU type here (e.g. g6e.12xlarge) and submit with --num-gpus 4.
   * Default g6e.4xlarge (1×L40S 48 GB) → g5.4xlarge (1×A10G 24 GB).
   */
  readonly instanceTypes?: string[];
  /**
   * Max vCPUs the CE may scale to. Caps concurrent jobs. Default 16 (one g6e.4xlarge).
   * Raise to run several RL jobs in parallel or to allow a larger multi-GPU instance.
   */
  readonly maxvCpus?: number;
  /** Email to notify on Batch job terminal state (opt-in). See PatternBStack. */
  readonly notifyEmail?: string;
  /** Job-name prefix the notification rule filters on. Default 'isaac-rl-'. */
  readonly notifyJobNamePrefix?: string;
  /** Job timeout. Default 12 h (RL PPO runs longer than a small IL fine-tune). */
  readonly timeout?: cdk.Duration;
  /**
   * Use Spot capacity (default true). Spot is the cost-optimal default and the
   * verified behavior, but a Spot reclaim restarts the job (the bootstrap resumes from
   * EFS — yet a reclaim *before* the first checkpoint replays the ~5-min Isaac Sim boot
   * each time). Set false for an On-Demand CE when a run must complete reliably without
   * reclaim — e.g. a first E2E smoke, or a short run that has no checkpoint to resume.
   * On-Demand uses BEST_FIT_PROGRESSIVE (the managed-EC2 On-Demand default).
   */
  readonly useSpot?: boolean;
}

/** Container resource ask. Must fit the SMALLEST instance in the fallback list
 *  (g5.4xlarge: 16 vCPU / 64 GiB). Isaac Sim headless is host-RAM hungry; 48 GiB
 *  leaves headroom under 64 while still fitting the fallback. */
const JOB_VCPUS = 16;
const JOB_MEMORY_GIB = 48;

export class RlPatternAStack extends cdk.Stack {
  public readonly computeEnvironment: batch.ManagedEc2EcsComputeEnvironment;
  public readonly jobQueue: batch.JobQueue;
  public readonly jobDefinition: batch.EcsJobDefinition;
  /** The role the container assumes — its AWS creds for S3 artifact I/O. */
  public readonly jobRole: iam.Role;
  public readonly notifications?: TrainingNotifications;

  constructor(scope: Construct, id: string, props: RlPatternAStackProps) {
    super(scope, id, props);

    const { base } = props;
    const namePrefix = props.namePrefix ?? 'pai';
    const instanceTypeStrings = props.instanceTypes ?? ['g6e.4xlarge', 'g5.4xlarge'];

    // --- Security group: egress-all (ECR/S3/NGC over NAT); EFS ingress added below. ---
    const batchSg = new ec2.SecurityGroup(this, 'BatchSg', {
      vpc: base.vpc,
      allowAllOutbound: true,
      description: 'PAI RL Pattern A Batch GPU instances',
    });
    // Let the Batch instances mount the shared EFS (NFS 2049). A standalone ingress
    // resource *here* (the consuming stack) pointing at base's EFS SG keeps the stack
    // dependency one-directional (RL-A → Base); mutating base's SG via connections
    // would form a cycle. Same verified pattern as IL Pattern A.
    new ec2.CfnSecurityGroupIngress(this, 'EfsIngressFromBatch', {
      groupId: base.fileSystem.connections.securityGroups[0].securityGroupId,
      ipProtocol: 'tcp',
      fromPort: 2049,
      toPort: 2049,
      sourceSecurityGroupId: batchSg.securityGroupId,
      description: 'RL Pattern A Batch jobs mount shared EFS (NFS)',
    });

    // --- Instance role (EC2 trust): the ECS agent pulls our image + registers. ---
    const instanceRole = new iam.Role(this, 'InstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      description: 'PAI RL Pattern A Batch EC2 instance role (ECS agent: image pull, register)',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEC2ContainerServiceforEC2Role'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // --- Execution role (ecs-tasks trust): log driver + ECR auth for the task. ---
    const executionRole = new iam.Role(this, 'ExecutionRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'PAI RL Pattern A Batch task execution role (awslogs + ECR auth)',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
    });

    // --- Job role (ecs-tasks trust): the container's AWS creds for S3 I/O. ---
    // jobBasePolicy = read dataBucket, read/write artifactBucket, pull ECR. RL only
    // writes the trained policy/ONNX to the artifact bucket; the simulator generates
    // its own experience, so there is no dataset read to grant beyond the base policy.
    this.jobRole = new iam.Role(this, 'JobRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'PAI RL Pattern A Batch job role - S3 artifact write for the Isaac Lab RL container',
      managedPolicies: [base.jobBasePolicy],
    });

    // --- Launch template: enlarge the root volume. The isaac-sim image is large
    // (~20 GB) plus the Isaac Lab install; 250 GB gp3 (encrypted) leaves headroom.
    // Checkpoints/logs go to EFS, not this disk. No ImageId, so Batch still selects
    // its GPU AMI (ECS_AL2_NVIDIA) automatically. ---
    const launchTemplate = new ec2.LaunchTemplate(this, 'BatchLaunchTemplate', {
      requireImdsv2: true, // enforce IMDSv2 (token-required) on the Batch GPU hosts
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(250, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
            deleteOnTermination: true,
          }),
        },
      ],
    });

    // --- Compute Environment: managed EC2, single-GPU g6e→g5 fallback. ---
    // Spot by default (cost-optimal, allocationStrategy SPOT_PRICE_CAPACITY_OPTIMIZED);
    // useSpot:false gives an On-Demand CE for reclaim-free runs (first E2E / short runs
    // with no checkpoint to resume).
    const useSpot = props.useSpot ?? true;
    this.computeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(this, 'GpuComputeEnv', {
      vpc: base.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [batchSg],
      instanceTypes: instanceTypeStrings.map((t) => new ec2.InstanceType(t)),
      useOptimalInstanceClasses: false, // we name exact GPU types; don't add C/M/R families
      spot: useSpot, // Spot → SPOT_PRICE_CAPACITY_OPTIMIZED; On-Demand → BEST_FIT_PROGRESSIVE
      maxvCpus: props.maxvCpus ?? 16,
      instanceRole,
      launchTemplate,
    });

    this.jobQueue = new batch.JobQueue(this, 'GpuJobQueue', {
      computeEnvironments: [{ computeEnvironment: this.computeEnvironment, order: 1 }],
    });

    // --- Job Definition: the Isaac Lab RL image; command = the RL Batch bootstrap. ---
    // rl_train_bootstrap.py is read at synth and injected via `python3 -c "<stub>"`, so
    // the image stays generic/task-agnostic and is never rebuilt to carry glue. The
    // bootstrap is stdlib + boto3 (boto3 is baked into the system python by the
    // Dockerfile).
    //
    // The injected command is shipped through Batch's ECS RunTask container-override
    // channel, which has a hard 8192-byte ceiling on the whole overrides blob. The raw
    // RL bootstrap is ~10.6 KB — over the limit (IL Pattern A's 5.7 KB bootstrap fit, so
    // the naive `python3 -c "<raw src>"` only failed for RL, and only at real submit).
    // We therefore zlib-compress the source at synth (~5.3 KB) and inject a tiny
    // decompress-and-run stub. zlib's DEFLATE stream is interoperable (Node deflate ->
    // Python zlib.decompress, verified), and exec under a globals dict with
    // __name__='__main__' fires the bootstrap's `if __name__ == "__main__": main()`.
    const bootstrapSrc = fs.readFileSync(
      path.join(__dirname, '..', '..', 'containers', 'isaac-lab-rl', 'rl_train_bootstrap.py'),
      'utf8',
    );
    const packedBootstrap = zlib.deflateSync(Buffer.from(bootstrapSrc, 'utf8'), { level: 9 }).toString('base64');
    const bootstrapStub =
      'import base64,zlib;' +
      `exec(compile(zlib.decompress(base64.b64decode("${packedBootstrap}")),"rl_train_bootstrap.py","exec"),` +
      '{"__name__":"__main__"})';

    const efsVolume = batch.EcsVolume.efs({
      name: 'shared-efs',
      fileSystem: base.fileSystem,
      containerPath: '/mnt/efs',
      enableTransitEncryption: true,
    });

    const container = new batch.EcsEc2ContainerDefinition(this, 'IsaacRlContainer', {
      image: ecs.ContainerImage.fromEcrRepository(base.isaacLabRlRepo, 'latest'),
      cpu: JOB_VCPUS,
      memory: cdk.Size.gibibytes(JOB_MEMORY_GIB),
      gpu: 1,
      jobRole: this.jobRole,
      executionRole,
      command: ['python3', '-c', bootstrapStub],
      volumes: [efsVolume],
      // Static defaults; per-job values (task, output, GPUs) arrive as container-override
      // environment at SubmitJob time (see rl_launch.py). Defaults make the JobDefinition
      // runnable as-is for the verified reference task with no overrides.
      environment: {
        RL_TASK: 'Isaac-Velocity-Rough-H1-v0',
        RL_EXPERIMENT_NAME: 'h1_rough',
        RL_OUTPUT_S3: `s3://${base.artifactBucket.bucketName}/isaac-rl`,
        RL_CHECKPOINT_DIR: '/mnt/efs/rl-checkpoints',
        RL_NUM_GPUS: '1',
        RL_ISAACLAB_DIR: '/workspace/IsaacLab',
      },
    });

    this.jobDefinition = new batch.EcsJobDefinition(this, 'IsaacRlJobDef', {
      container,
      // Spot reclaim → one retry; the bootstrap resumes from the EFS rsl_rl run dir.
      retryAttempts: 2,
      // Retry strategies are evaluated in order; the first match wins, else the job
      // retries up to retryAttempts. The bootstrap's fail-fast liveness guard exits 42
      // when the trainer booted but never started learning (the 5.5 h idle-run class) —
      // that is a deterministic image/config fault, so EXIT (no retry): retrying would
      // just burn another full deadline. A genuine Spot reclaim still RETRYs so the
      // resume-from-EFS path runs.
      retryStrategies: [
        batch.RetryStrategy.of(batch.Action.EXIT, batch.Reason.custom({ onExitCode: '42' })),
        batch.RetryStrategy.of(batch.Action.RETRY, batch.Reason.SPOT_INSTANCE_RECLAIMED),
      ],
      timeout: props.timeout ?? cdk.Duration.hours(12),
    });

    // --- Job-completion notifications (opt-in) — Batch job-state-change rule. ---
    if (props.notifyEmail) {
      this.notifications = new TrainingNotifications(this, 'Notifications', {
        namePrefix,
        notifyEmail: props.notifyEmail,
      });
      this.notifications.addBatchJobRule(this.jobQueue.jobQueueArn, props.notifyJobNamePrefix ?? 'isaac-rl-');

      new cdk.CfnOutput(this, 'NotificationTopicArn', {
        value: this.notifications.topic.topicArn,
        description: 'SNS topic for Batch job terminal-state notifications',
      });
    }

    // --- Outputs: everything rl_launch.py needs. ---
    new cdk.CfnOutput(this, 'JobQueueArn', {
      value: this.jobQueue.jobQueueArn,
      description: 'Pass to rl_launch.py --job-queue',
    });
    new cdk.CfnOutput(this, 'JobDefinitionArn', {
      value: this.jobDefinition.jobDefinitionArn,
      description: 'Pass to rl_launch.py --job-definition',
    });
    new cdk.CfnOutput(this, 'ImageUriHint', {
      value: `${base.isaacLabRlRepo.repositoryUri}:latest`,
      description: 'Build target for containers/isaac-lab-rl/build.sh',
    });
    new cdk.CfnOutput(this, 'OutputS3Hint', {
      value: `s3://${base.artifactBucket.bucketName}/isaac-rl`,
      description: 'Trained policy + ONNX land at <this>/<job>/output/<run>/',
    });
  }
}
