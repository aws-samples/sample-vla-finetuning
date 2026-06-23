/**
 * PatternAStack — IL axis, Pattern A (AWS Batch + g6e Spot).
 *
 * Pattern A is the "small / fast / cheap" IL backend in the decision rule table
 * (ARCHITECTURE §3.1: est. VRAM ≤ 48 GB and wall-clock ≤ ~2–4 h → Batch Spot on a
 * single-GPU L40S). Unlike Pattern B (SageMaker-managed), Batch places GPU capacity
 * in **our** VPC, so this stack owns the Compute Environment / Job Queue / Job
 * Definition — exactly the CE/JQ/JD automation both source assets left manual
 * (sample-embodied-ai-platform's console-deploy path; the Isaac Lab infra template's
 * `batch-infra.ts` left it out of scope). It runs the **same verified `vla-ft` container**
 * as Pattern B — the container lock is untouched.
 *
 * Backend portability problem this stack solves: the vla-ft image is BYOC for
 * SageMaker — it ships `src/train.py` as a SageMaker `source_dir` (not baked into the
 * image) and relies on SageMaker for dataset download + model upload. Batch has none
 * of that. So the Job Definition's command is a tiny self-contained bootstrap
 * (`containers/vla-ft/batch_bootstrap.py`, injected via `python3 -c` at synth — the
 * image is never rebuilt) that stages dataset S3→local, fetches the unchanged
 * train.py from S3, runs it, and uploads the model to S3 in SageMaker's exact output
 * layout. train.py already supports this: it reads `SM_HP_*` env (its env fallback),
 * and `SM_CHANNEL_TRAINING` / `SM_MODEL_DIR` / `SM_NUM_GPUS` are env-overridable.
 *
 * Capacity strategy: a managed Spot CE with `SPOT_PRICE_CAPACITY_OPTIMIZED` (the L2
 * default for Spot) across a g6e→g5 single-GPU instance fallback IS Batch's native
 * answer to insufficient-capacity — strictly better here than the trial-RunInstances
 * AzSelector (which exists for one-shot EC2/DCV launches, not a Batch CE that already
 * draws from a pool). So Pattern A does NOT wire AzSelector.
 *
 * Spot vs On-Demand is PER JOB, not per deploy: this stack owns BOTH a Spot CE/queue
 * (the cheap default) and an On-Demand CE/queue (reclaim-free for a long sanctioned
 * run), sharing one Job Definition. The launcher picks the queue at SubmitJob time via
 * the plan's `spot` flag — no redeploy to switch. Each CE scales to 0 vCPUs when idle,
 * so the waiting queue carries no standing cost; only a running job is billed.
 *
 * Checkpoints land on the shared EFS (`/mnt/efs/...`), so a Spot reclaim + Batch
 * retry resumes via train.py's find_latest_checkpoint(). Synthesizes with no
 * credentials. Standing cost is the base stack's NAT; the CE scales to 0 vCPUs when
 * idle, so GPU cost is incurred only while a job runs.
 */
import * as cdk from 'aws-cdk-lib';
import * as fs from 'fs';
import * as path from 'path';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as batch from 'aws-cdk-lib/aws-batch';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import { Construct } from 'constructs';
import { SharedBaseStack } from '../shared/base-stack';
import { TrainingNotifications } from '../shared/notifications';

export interface PatternAStackProps extends cdk.StackProps {
  /** The shared base stack whose VPC, EFS, ECR repo, buckets, and jobBasePolicy this imports. */
  readonly base: SharedBaseStack;
  /** Prefix for physical resource names (must match the base stack's). Default 'pai'. */
  readonly namePrefix?: string;
  /**
   * Single-GPU instance fallback for the Compute Environment, highest priority
   * first. Pattern A is the single-GPU tier (rule table), so these are all 1-GPU
   * types; Spot SPOT_PRICE_CAPACITY_OPTIMIZED picks the cheapest available. The Job
   * Definition's memory/cpu must fit the SMALLEST of these. Default g6e.4xlarge
   * (1×L40S 48 GB, quota secured) → g5.4xlarge (1×A10G 24 GB) fallback.
   */
  readonly instanceTypes?: string[];
  /**
   * Instance fallback for the ON-DEMAND CE only. The OD queue is the sanctioned
   * long-run path (spot=false), used for full-VLM fine-tunes whose ~40 GB replica
   * does NOT fit a 24 GB A10G — so the OD CE must be L40S-only (no g5 fallback), or
   * Batch's BEST_FIT_PROGRESSIVE will silently place the job on whichever GPU has
   * capacity and the run OOMs at step 0 in the vision tower. Default ['g6e.4xlarge']
   * (1×L40S 48 GB). The Spot CE keeps the g6e→g5 fallback (instanceTypes) for cheap /
   * short / expert-only runs that DO fit 24 GB.
   */
  readonly onDemandInstanceTypes?: string[];
  /**
   * Max vCPUs the CE may scale to. Caps concurrent jobs (16 vCPU/instance → 16 = one
   * g6e.4xlarge at a time). Default 16. Raise to run several fine-tunes in parallel.
   */
  readonly maxvCpus?: number;
  /**
   * Extra S3 bucket ARNs the job role may READ datasets from, beyond the platform
   * dataBucket. Mirrors PatternBStack — e.g. the verified openarm-lift dataset bucket.
   * Read-only.
   */
  readonly extraDatasetReadArns?: string[];
  /** Email to notify on Batch job terminal state (opt-in). See PatternBStack. */
  readonly notifyEmail?: string;
  /** Job-name prefix the notification rule filters on. Default 'vla-ft-'. */
  readonly notifyJobNamePrefix?: string;
  /**
   * Per-attempt wall-clock ceiling for the Job Definition. Batch SIGKILLs the job
   * (exit 137) at this limit regardless of training health. Must cover the SLOWEST
   * sanctioned run: a single-L40S full-VLM LoRA fine-tune (vision unfrozen, 20000 steps,
   * batch 16). MEASURED, not estimated: job ...151444 attempt ...b62f0794 reached only
   * step 19000/20000 at the 18 h (64800 s) ceiling before the timeout SIGKILL — i.e.
   * ~3.36 s/step on a single L40S (≫ the 1.0 s/step multi-GPU expert-only anchor in
   * vla_ft_decide), so a clean 20000-step run needs ~19 h + bootstrap (base-weight
   * download, dataset sync) + the final ~9 GB S3 model upload. The earlier "~12.8 h"
   * sizing was wrong for this regime — that is why healthy runs kept dying one step
   * short. Default 28 h covers the measured ~19 h with generous headroom. EFS-resume
   * (same job_name) still continues an interrupted run, but a single uninterrupted
   * attempt must now be able to finish on its own.
   */
  readonly attemptTimeout?: cdk.Duration;
}

/** Container resource ask. Must fit the SMALLEST instance in the fallback list
 *  (g5.4xlarge: 16 vCPU / 64 GiB). GPU host RAM beyond this is unused headroom. */
const JOB_VCPUS = 16;
const JOB_MEMORY_GIB = 48;

export class PatternAStack extends cdk.Stack {
  /** Spot Compute Environment (the cheap default tier). */
  public readonly computeEnvironment: batch.ManagedEc2EcsComputeEnvironment;
  /** On-Demand Compute Environment (reclaim-free tier for long sanctioned runs). */
  public readonly computeEnvironmentOnDemand: batch.ManagedEc2EcsComputeEnvironment;
  /** Spot Job Queue — the default (spot=true at submit). `JobQueueArn` output. */
  public readonly jobQueue: batch.JobQueue;
  /** On-Demand Job Queue — selected per-job (spot=false). `JobQueueArnOnDemand` output. */
  public readonly jobQueueOnDemand: batch.JobQueue;
  /** Shared Job Definition — both queues draw from it (only the CE capacity type differs). */
  public readonly jobDefinition: batch.EcsJobDefinition;
  /** The role the container assumes — its AWS creds for S3 dataset/model I/O. */
  public readonly jobRole: iam.Role;
  public readonly notifications?: TrainingNotifications;

  constructor(scope: Construct, id: string, props: PatternAStackProps) {
    super(scope, id, props);

    const { base } = props;
    const namePrefix = props.namePrefix ?? 'pai';
    const instanceTypeStrings = props.instanceTypes ?? ['g6e.4xlarge', 'g5.4xlarge'];
    // On-Demand = sanctioned long-run / full-VLM path → L40S-only (no 24 GB g5 fallback,
    // which OOMs a ~40 GB full-VLM replica at step 0). Spot keeps the g6e→g5 fallback.
    const onDemandInstanceTypeStrings = props.onDemandInstanceTypes ?? ['g6e.4xlarge'];

    // --- Security group: egress-all (ECR/S3/HF over NAT); EFS ingress added below. ---
    const batchSg = new ec2.SecurityGroup(this, 'BatchSg', {
      vpc: base.vpc,
      allowAllOutbound: true,
      description: 'PAI Pattern A Batch GPU instances',
    });
    // Let the Batch instances mount the shared EFS (NFS 2049). We must NOT mutate
    // base's EFS SG via connections.allowDefaultPortFrom(batchSg): that would add an
    // ingress rule *in the base stack* referencing this stack's SG, and since this
    // stack already imports base's VPC, the two stacks would form a dependency cycle.
    // Instead create a standalone ingress resource *here* (consuming stack) pointing
    // at base's EFS SG — the dependency stays one-directional (A → Base). This is the
    // verified Isaac Lab infra pattern ("separate resource to avoid a
    // circular reference").
    new ec2.CfnSecurityGroupIngress(this, 'EfsIngressFromBatch', {
      groupId: base.fileSystem.connections.securityGroups[0].securityGroupId,
      ipProtocol: 'tcp',
      fromPort: 2049,
      toPort: 2049,
      sourceSecurityGroupId: batchSg.securityGroupId,
      description: 'Pattern A Batch jobs mount shared EFS (NFS)',
    });

    // --- Instance role (EC2 trust): the ECS agent pulls our image + registers. ---
    const instanceRole = new iam.Role(this, 'InstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      description: 'PAI Pattern A Batch EC2 instance role (ECS agent: image pull, register)',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEC2ContainerServiceforEC2Role'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // --- Execution role (ecs-tasks trust): log driver + ECR auth for the task. ---
    const executionRole = new iam.Role(this, 'ExecutionRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'PAI Pattern A Batch task execution role (awslogs + ECR auth)',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
    });

    // --- Job role (ecs-tasks trust): the container's AWS creds for S3 I/O. ---
    // Same envelope as Pattern B's execution role: jobBasePolicy (read dataBucket,
    // read/write artifactBucket, pull ECR) + optional extra dataset read.
    this.jobRole = new iam.Role(this, 'JobRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'PAI Pattern A Batch job role - S3 dataset read + model write for the vla-ft container',
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

    // --- Launch template: enlarge the root volume. The default ECS GPU AMI root is
    // too small for the ~25 GB uncompressed image + base weights + dataset + staged
    // model. 200 GB gp3 (encrypted); checkpoints go to EFS, not this disk. No ImageId
    // here, so Batch still selects its GPU AMI (ECS_AL2_NVIDIA) automatically. ---
    const launchTemplate = new ec2.LaunchTemplate(this, 'BatchLaunchTemplate', {
      requireImdsv2: true, // enforce IMDSv2 (token-required) on the Batch GPU hosts
      blockDevices: [
        {
          deviceName: '/dev/xvda',
          volume: ec2.BlockDeviceVolume.ebs(200, {
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            encrypted: true,
            deleteOnTermination: true,
          }),
        },
      ],
    });

    // --- Dual Compute Environment + Job Queue: Spot (default) AND On-Demand, both live. ---
    // Spot vs On-Demand is a Compute-Environment property, so it can't be flipped per job
    // on one CE. The standard answer is to deploy BOTH and select the queue at SubmitJob
    // time. Each CE scales to 0 vCPUs when idle (no instance runs until a job lands), so the
    // waiting queue costs nothing — making per-job switching free of any standing cost and,
    // crucially, requiring NO redeploy to move a job between Spot and On-Demand.
    //
    //   - Spot CE → allocationStrategy SPOT_PRICE_CAPACITY_OPTIMIZED (the cheap default;
    //     a reclaim RETRYs and train.py resumes from the EFS checkpoint).
    //   - On-Demand CE → BEST_FIT_PROGRESSIVE (reclaim-free, for a long sanctioned run that
    //     should not eat Spot-reclaim restarts — the same need grootUseSpot:false serves).
    const ceCommon = {
      vpc: base.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [batchSg],
      instanceTypes: instanceTypeStrings.map((t) => new ec2.InstanceType(t)),
      useOptimalInstanceClasses: false, // we name exact GPU types; don't add C/M/R families
      maxvCpus: props.maxvCpus ?? 16,
      instanceRole,
      launchTemplate,
    };
    this.computeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(this, 'GpuComputeEnv', {
      ...ceCommon,
      spot: true, // SPOT_PRICE_CAPACITY_OPTIMIZED
    });
    this.computeEnvironmentOnDemand = new batch.ManagedEc2EcsComputeEnvironment(this, 'GpuComputeEnvOnDemand', {
      ...ceCommon,
      // L40S-only override: a full-VLM run routed here (spot=false) must not land on a
      // 24 GB g5 — BEST_FIT_PROGRESSIVE would otherwise pick whatever has capacity.
      instanceTypes: onDemandInstanceTypeStrings.map((t) => new ec2.InstanceType(t)),
      spot: false, // BEST_FIT_PROGRESSIVE (On-Demand)
    });

    this.jobQueue = new batch.JobQueue(this, 'GpuJobQueue', {
      computeEnvironments: [{ computeEnvironment: this.computeEnvironment, order: 1 }],
    });
    this.jobQueueOnDemand = new batch.JobQueue(this, 'GpuJobQueueOnDemand', {
      computeEnvironments: [{ computeEnvironment: this.computeEnvironmentOnDemand, order: 1 }],
    });

    // --- Job Definition: the verified vla-ft image; command = the Batch bootstrap. ---
    // batch_bootstrap.py is read at synth and embedded as `python3 -c "<src>"`, so the
    // image stays byte-identical and train.py stays S3-iterable (VLA_FT_CODE_S3). The
    // bootstrap is stdlib + boto3 (boto3 is in the image via sagemaker-training).
    const bootstrapSrc = fs.readFileSync(
      path.join(__dirname, '..', '..', 'containers', 'vla-ft', 'batch_bootstrap.py'),
      'utf8',
    );

    const efsVolume = batch.EcsVolume.efs({
      name: 'shared-efs',
      fileSystem: base.fileSystem,
      containerPath: '/mnt/efs',
      enableTransitEncryption: true,
    });

    const container = new batch.EcsEc2ContainerDefinition(this, 'VlaFtContainer', {
      image: ecs.ContainerImage.fromEcrRepository(base.vlaFtRepo, 'latest'),
      cpu: JOB_VCPUS,
      memory: cdk.Size.gibibytes(JOB_MEMORY_GIB),
      gpu: 1,
      jobRole: this.jobRole,
      executionRole,
      command: ['python3', '-c', bootstrapSrc],
      volumes: [efsVolume],
      // Static defaults; per-job values (dataset, hyperparameters, job name) arrive as
      // container-override environment at SubmitJob time (see batch_launch.py).
      environment: {
        VLA_FT_CODE_S3: `s3://${base.artifactBucket.bucketName}/vla-ft-code/train.py`,
        VLA_FT_OUTPUT_S3: `s3://${base.artifactBucket.bucketName}/vla-ft`,
        SM_NUM_GPUS: '1',
        VLA_FT_CHECKPOINT_DIR: '/mnt/efs/checkpoints',
        PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True',
      },
    });

    this.jobDefinition = new batch.EcsJobDefinition(this, 'VlaFtJobDef', {
      container,
      // Spot reclaim → retries; train.py resumes from the EFS checkpoint (same job_name).
      retryAttempts: 2,
      // Sized for the MEASURED slowest sanctioned run: single-L40S full-VLM LoRA
      // (~3.36 s/step × 20000 ≈ 19 h + bootstrap + ~9 GB upload). The old 18 h ceiling
      // SIGKILLed a healthy run at step 19000/20000 (job ...151444 / b62f0794), losing
      // the S3 upload (the bootstrap only uploads on a clean exit). 28 h gives headroom.
      // See attemptTimeout prop.
      timeout: props.attemptTimeout ?? cdk.Duration.hours(28),
    });

    // --- Job-completion notifications (opt-in) — Batch job-state-change rule. ---
    // The topic is owned by Base (one per platform); we only add this stack's rule, so
    // multiple notifying stacks no longer collide on the fixed topic name.
    if (base.notificationTopic) {
      this.notifications = new TrainingNotifications(this, 'Notifications', {
        topic: base.notificationTopic,
      });
      // Cover BOTH queues — a vla-ft- job may land on either depending on the per-job
      // spot/on-demand selection. The name-prefix filter is the same on both.
      this.notifications.addBatchJobRule(
        [this.jobQueue.jobQueueArn, this.jobQueueOnDemand.jobQueueArn],
        props.notifyJobNamePrefix ?? 'vla-ft-',
      );
    }

    // --- Outputs: everything batch_launch.py needs. ---
    new cdk.CfnOutput(this, 'JobQueueArn', {
      value: this.jobQueue.jobQueueArn,
      description: 'Spot queue (the default, spot=true). Pass to batch_launch.py --job-queue',
    });
    new cdk.CfnOutput(this, 'JobQueueArnOnDemand', {
      value: this.jobQueueOnDemand.jobQueueArn,
      description: 'On-Demand queue (spot=false, reclaim-free for long runs). Selected per-job.',
    });
    new cdk.CfnOutput(this, 'JobDefinitionArn', {
      value: this.jobDefinition.jobDefinitionArn,
      description: 'Pass to batch_launch.py --job-definition',
    });
    new cdk.CfnOutput(this, 'ImageUriHint', {
      value: `${base.vlaFtRepo.repositoryUri}:latest`,
      description: 'Build target for containers/vla-ft/build.sh (same image as Pattern B)',
    });
    new cdk.CfnOutput(this, 'CodeS3Hint', {
      value: `s3://${base.artifactBucket.bucketName}/vla-ft-code/train.py`,
      description: 'batch_launch.py uploads src/train.py here before each submit (source_dir equivalent)',
    });
    new cdk.CfnOutput(this, 'OutputS3Hint', {
      value: `s3://${base.artifactBucket.bucketName}/vla-ft`,
      description: 'Model artifacts land at <this>/<job>/output/ (SageMaker-equivalent layout)',
    });
    new cdk.CfnOutput(this, 'DataBucketName', {
      value: base.dataBucket.bucketName,
      description: 'Upload LeRobot v3 datasets here; reference as batch_launch.py --dataset-s3',
    });
  }
}
