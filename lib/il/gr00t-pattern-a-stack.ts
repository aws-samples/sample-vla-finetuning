/**
 * GrootPatternAStack — IL axis, GR00T N1.7 fine-tune (AWS Batch + g6e On-Demand).
 *
 * The platform's third IL engine, alongside the vla-ft PatternA/B (lerobot π0.5) stacks.
 * It fine-tunes NVIDIA GR00T N1.7 (a 3B VLA, Cosmos-Reason2-2B backbone) on a LeRobot
 * v2.1 dataset and emits a full merged HF checkpoint that `run_gr00t_server.py` loads for
 * sim rollout. It mirrors the IL/RL Pattern A skeleton — this stack OWNS the Batch
 * Compute Environment / Job Queue / Job Definition — running the `gr00t-n17` container.
 *
 * What's different from the vla-ft / RL Pattern A stacks (and WHY):
 *  - Image is the GR00T container (`base.grootRepo`).
 *  - The job command is `gr00t_train_bootstrap.py`, injected via `python3 -c "<zlib
 *    stub>"` at synth so the image is never rebuilt to carry glue. The raw bootstrap
 *    exceeds Batch's 8192-byte ECS container-override ceiling (the same trap RL hit), so
 *    it is zlib-compressed at synth and a tiny decompress-and-run stub is injected.
 *
 *  - **g6e (L40S 48 GB) only — NO g5 fallback.** GR00T N1.7 is a 3B Cosmos model; its
 *    own hardware doc requires 40 GB+ VRAM and the default frozen-backbone fine-tune peaks
 *    at ~35 GB/GPU. A10G 24 GB (g5) OOMs, so a g5 fallback would only schedule a job that
 *    is guaranteed to fail. The single-GPU tier is g6e.4xlarge (1×L40S, quota secured).
 *  - **On-Demand by default** (`useSpot` false). This is a single, multi-hour fine-tune
 *    with no EFS-resume wired (checkpoints land on the large local root volume, not EFS),
 *    so a Spot reclaim would restart from scratch — reclaim-free is the right default for
 *    a first run / the "one video" goal. Spot is a documented opt-in once the path is
 *    proven.
 *  - **No EFS mount.** The bootstrap keeps the dataset + base model + checkpoints on a
 *    300 GB local root volume (the 3B base is ~6 GB and full merged checkpoints are ~6 GB
 *    each). Not wiring EFS removes the cross-stack SG ingress this stack would otherwise
 *    carry unused.
 *  - The "intent" is a dataset + an embodiment tag, so the job role gets a dataset read on
 *    the GR00T-G1 bucket (`extraDatasetReadArns`), like Pattern B. The HF token (the base
 *    is ungated, so it is optional) is read launcher-side and injected as job env — never
 *    on the JobDefinition — so no SSM read on the job role.
 *
 * Synthesizes with no credentials; the CE scales to 0 vCPUs when idle, so GPU cost is
 * incurred only while a job runs. The bootstrap's fail-fast liveness guard makes the
 * booted-but-idle class (the 5.5 h burn) a fast exit-42, mapped here to no-retry.
 *
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

export interface GrootPatternAStackProps extends cdk.StackProps {
  /** The shared base stack whose VPC, ECR repo, buckets, and jobBasePolicy this imports. */
  readonly base: SharedBaseStack;
  /** Prefix for physical resource names (must match the base stack's). Default 'pai'. */
  readonly namePrefix?: string;
  /**
   * GPU instance fallback for the Compute Environment, highest priority first. GR00T N1.7
   * needs 40 GB+ VRAM (frozen-backbone peak ~35 GB), so these MUST be L40S/A100-class —
   * NO A10G (g5) fallback (it would OOM). Default g6e.4xlarge (1×L40S 48 GB, quota
   * secured). For a multi-GPU run, deploy with a 4-GPU type (e.g. g6e.12xlarge) and submit
   * with --num-gpus 4.
   */
  readonly instanceTypes?: string[];
  /**
   * Max vCPUs the CE may scale to. Caps concurrent jobs (16 vCPU/instance → 16 = one
   * g6e.4xlarge at a time). Default 16. Raise to run several fine-tunes in parallel or to
   * allow a larger multi-GPU instance.
   */
  readonly maxvCpus?: number;
  /**
   * Extra S3 bucket ARNs the job role may READ datasets from, beyond the platform
   * dataBucket. Mirrors PatternBStack — e.g. the GR00T-G1 LeRobot dataset bucket.
   * Read-only.
   */
  readonly extraDatasetReadArns?: string[];
  /** Email to notify on Batch job terminal state (opt-in). See PatternBStack. */
  readonly notifyEmail?: string;
  /** Job-name prefix the notification rule filters on. Default 'gr00t-n17-'. */
  readonly notifyJobNamePrefix?: string;
  /** Job timeout. Default 8 h (a frozen-backbone 3B fine-tune on a single L40S). */
  readonly timeout?: cdk.Duration;
  /**
   * Use Spot capacity (default false → On-Demand). On-Demand is the reclaim-free default
   * for this single long fine-tune: no EFS-resume is wired, so a Spot reclaim would
   * restart from scratch (re-download the 6 GB base + replay all steps). Set true only
   * once you accept that a reclaim repeats lost work. On-Demand uses
   * BEST_FIT_PROGRESSIVE; Spot uses SPOT_PRICE_CAPACITY_OPTIMIZED.
   */
  readonly useSpot?: boolean;
}

/** Container resource ask. g6e.4xlarge = 16 vCPU / 128 GiB; ask for the full vCPU and a
 *  generous slice of RAM (the GR00T dataloader + 3B model are memory-hungry) while leaving
 *  headroom for the ECS agent. */
const JOB_VCPUS = 16;
const JOB_MEMORY_GIB = 100;

export class GrootPatternAStack extends cdk.Stack {
  public readonly computeEnvironment: batch.ManagedEc2EcsComputeEnvironment;
  public readonly jobQueue: batch.JobQueue;
  public readonly jobDefinition: batch.EcsJobDefinition;
  /** The role the container assumes — its AWS creds for S3 dataset/model I/O. */
  public readonly jobRole: iam.Role;
  public readonly notifications?: TrainingNotifications;

  constructor(scope: Construct, id: string, props: GrootPatternAStackProps) {
    super(scope, id, props);

    const { base } = props;
    const namePrefix = props.namePrefix ?? 'pai';
    // L40S only — A10G 24 GB OOMs a 3B Cosmos model (no g5 fallback by design).
    const instanceTypeStrings = props.instanceTypes ?? ['g6e.4xlarge'];

    // --- Security group: egress-all (ECR / S3 / HuggingFace over NAT). ---
    const batchSg = new ec2.SecurityGroup(this, 'BatchSg', {
      vpc: base.vpc,
      allowAllOutbound: true,
      description: 'PAI GR00T Pattern A Batch GPU instances',
    });

    // --- Instance role (EC2 trust): the ECS agent pulls our image + registers. ---
    const instanceRole = new iam.Role(this, 'InstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      description: 'PAI GR00T Pattern A Batch EC2 instance role (ECS agent: image pull, register)',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEC2ContainerServiceforEC2Role'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // --- Execution role (ecs-tasks trust): log driver + ECR auth for the task. ---
    const executionRole = new iam.Role(this, 'ExecutionRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'PAI GR00T Pattern A Batch task execution role (awslogs + ECR auth)',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
    });

    // --- Job role (ecs-tasks trust): the container's AWS creds for S3 I/O. ---
    // Same envelope as Pattern B: jobBasePolicy (read dataBucket, read/write artifactBucket,
    // pull ECR) + optional extra dataset read (the GR00T-G1 LeRobot bucket).
    this.jobRole = new iam.Role(this, 'JobRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'PAI GR00T Pattern A Batch job role - S3 dataset read + checkpoint write for the gr00t-n17 container',
      managedPolicies: [base.jobBasePolicy],
    });
    if (props.extraDatasetReadArns && props.extraDatasetReadArns.length > 0) {
      this.jobRole.addToPolicy(
        new iam.PolicyStatement({
          sid: 'ExtraDatasetRead',
          actions: ['s3:GetObject', 's3:ListBucket', 's3:GetBucketLocation'],
          resources: props.extraDatasetReadArns.flatMap((arn) => [arn, `${arn}/*`]),
        }),
      );
    }

    // --- Launch template: enlarge the root volume. The image (~30 GB), the ~6 GB HF base
    // model, the dataset, and up to save_total_limit (5) full merged ~6 GB checkpoints all
    // live on the local root (no EFS). 300 GB gp3 (encrypted) leaves headroom. No ImageId,
    // so Batch still selects its GPU AMI (ECS_AL2_NVIDIA) automatically. ---
    const launchTemplate = new ec2.LaunchTemplate(this, 'BatchLaunchTemplate', {
      requireImdsv2: true, // enforce IMDSv2 (token-required) on the Batch GPU hosts
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(300, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
            deleteOnTermination: true,
          }),
        },
      ],
    });

    // --- Compute Environment: managed EC2, g6e L40S, On-Demand by default. ---
    const useSpot = props.useSpot ?? false;
    this.computeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(this, 'GpuComputeEnv', {
      vpc: base.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [batchSg],
      instanceTypes: instanceTypeStrings.map((t) => new ec2.InstanceType(t)),
      useOptimalInstanceClasses: false, // we name exact GPU types; don't add C/M/R families
      spot: useSpot, // On-Demand → BEST_FIT_PROGRESSIVE; Spot → SPOT_PRICE_CAPACITY_OPTIMIZED
      maxvCpus: props.maxvCpus ?? 16,
      instanceRole,
      launchTemplate,
    });

    this.jobQueue = new batch.JobQueue(this, 'GpuJobQueue', {
      computeEnvironments: [{ computeEnvironment: this.computeEnvironment, order: 1 }],
    });

    // --- Job Definition: the GR00T image; command = the GR00T Batch bootstrap. ---
    // gr00t_train_bootstrap.py is read at synth and injected via `python3 -c "<stub>"`, so
    // the image stays task-agnostic and is never rebuilt to carry glue. The raw bootstrap
    // exceeds Batch's 8192-byte ECS container-override ceiling, so we zlib-compress at
    // synth and inject a decompress-and-run stub (Node deflate ↔ Python zlib.decompress is
    // interoperable; exec under __name__='__main__' fires main()). Same trap/solution as
    // the RL Pattern A stack.
    const bootstrapSrc = fs.readFileSync(
      path.join(__dirname, '..', '..', 'containers', 'gr00t-n17', 'gr00t_train_bootstrap.py'),
      'utf8',
    );
    const packedBootstrap = zlib.deflateSync(Buffer.from(bootstrapSrc, 'utf8'), { level: 9 }).toString('base64');
    const bootstrapStub =
      'import base64,zlib;' +
      `exec(compile(zlib.decompress(base64.b64decode("${packedBootstrap}")),"gr00t_train_bootstrap.py","exec"),` +
      '{"__name__":"__main__"})';

    // --- /dev/shm sizing. The GR00T PyTorch DataLoader hands large multi-camera image
    // tensors (UNITREE_G1 ego_view 480x640x3, both arms) between worker processes via
    // shared-memory-backed file storage. The container default /dev/shm is 64 MiB, which
    // the loader blows past at step 0 ("Bus error ... dataloader's workers are out of shared
    // memory" / "No space left on device"). 16 GiB is ample and sits well under the 100 GiB
    // container memory on a g6e.4xlarge (128 GiB RAM). EC2 launch type only (we are). ---
    const linuxParameters = new batch.LinuxParameters(this, 'LinuxParams', {
      sharedMemorySize: cdk.Size.gibibytes(16),
    });

    const container = new batch.EcsEc2ContainerDefinition(this, 'GrootContainer', {
      image: ecs.ContainerImage.fromEcrRepository(base.grootRepo, 'latest'),
      cpu: JOB_VCPUS,
      memory: cdk.Size.gibibytes(JOB_MEMORY_GIB),
      gpu: 1,
      jobRole: this.jobRole,
      executionRole,
      linuxParameters,
      command: ['python3', '-c', bootstrapStub],
      // Static defaults; per-job values (dataset, output, steps, GPUs) arrive as
      // container-override environment at SubmitJob time (see gr00t_launch.py). Defaults
      // make the JobDefinition runnable as-is for the G1 reference dataset.
      environment: {
        GROOT_BASE_MODEL: 'nvidia/GR00T-N1.7-3B',
        GROOT_EMBODIMENT_TAG: 'UNITREE_G1',
        GROOT_OUTPUT_S3: `s3://${base.artifactBucket.bucketName}/gr00t-n17`,
        GROOT_NUM_GPUS: '1',
        GROOT_MAX_STEPS: '2000',
        GROOT_SAVE_STEPS: '1000',
        HF_HOME: '/opt/ml/hf-cache',
      },
    });

    this.jobDefinition = new batch.EcsJobDefinition(this, 'GrootJobDef', {
      container,
      retryAttempts: 2,
      // Retry strategies evaluated in order; first match wins. The liveness guard exits 42
      // when the trainer booted but never started learning (the idle-run class) — a
      // deterministic image/config fault, so EXIT (no retry): retrying just burns another
      // deadline. A Spot reclaim (only possible when useSpot:true) RETRYs (it restarts from
      // scratch since no EFS resume is wired — acceptable, it eventually completes).
      //
      retryStrategies: [
        batch.RetryStrategy.of(batch.Action.EXIT, batch.Reason.custom({ onExitCode: '42' })),
        batch.RetryStrategy.of(batch.Action.RETRY, batch.Reason.SPOT_INSTANCE_RECLAIMED),
      ],
      timeout: props.timeout ?? cdk.Duration.hours(8),
    });

    // --- Job-completion notifications (opt-in) — Batch job-state-change rule. ---
    if (base.notificationTopic) {
      this.notifications = new TrainingNotifications(this, 'Notifications', {
        topic: base.notificationTopic,
      });
      this.notifications.addBatchJobRule(this.jobQueue.jobQueueArn, props.notifyJobNamePrefix ?? 'gr00t-n17-');
    }

    // --- Outputs: everything gr00t_launch.py needs. ---
    new cdk.CfnOutput(this, 'JobQueueArn', {
      value: this.jobQueue.jobQueueArn,
      description: 'Pass to gr00t_launch.py --job-queue',
    });
    new cdk.CfnOutput(this, 'JobDefinitionArn', {
      value: this.jobDefinition.jobDefinitionArn,
      description: 'Pass to gr00t_launch.py --job-definition',
    });
    new cdk.CfnOutput(this, 'ImageUriHint', {
      value: `${base.grootRepo.repositoryUri}:latest`,
      description: 'Build target for containers/gr00t-n17/build.sh',
    });
    new cdk.CfnOutput(this, 'OutputS3Hint', {
      value: `s3://${base.artifactBucket.bucketName}/gr00t-n17`,
      description: 'Checkpoint (checkpoint-<N>/ + experiment_cfg/) lands at <this>/<job>/output/',
    });
  }
}
