/**
 * SharedBaseStack
 *
 * The infrastructure both axes (IL fine-tune, RL training) share: network, shared
 * filesystem, container registries, durable data/artifact buckets, and a base IAM
 * policy that training jobs attach to reach those resources. Pattern stacks (A/B/RL)
 * import these via the public readonly properties rather than re-creating them, so
 * VPC/EFS/ECR/S3 exist once per environment.
 *
 * Idiomatic L2 constructs throughout — unlike the source Isaac Lab infra template
 * (which used L1 Cfn* to mirror a hand-written template 1:1), this is greenfield so
 * we take the L2 defaults (sensible SGs, subnet routing, encryption).
 *
 * Synthesizes with no credentials and no GPU capacity probe (AzSelector is opt-in,
 * wired only by GPU-placing pattern stacks). Cost at this layer is ~0 until a
 * pattern stack runs a job: NAT Gateway hourly + EFS/S3 at-rest are the only
 * standing charges, and those are deferred to whichever stack instantiates the VPC.
 */
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as efs from 'aws-cdk-lib/aws-efs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface SharedBaseStackProps extends cdk.StackProps {
  /**
   * Prefix applied to physical resource names (buckets, repos) so multiple
   * platform deployments can coexist in one account/region. Default 'pai'.
   */
  readonly namePrefix?: string;
}

export class SharedBaseStack extends cdk.Stack {
  /** Platform VPC (2 AZ, public + private-with-egress). */
  public readonly vpc: ec2.Vpc;
  /** Shared EFS file system (checkpoints, datasets, scratch across jobs). */
  public readonly fileSystem: efs.FileSystem;
  /** Durable bucket for input datasets / demonstration data (RETAIN). */
  public readonly dataBucket: s3.Bucket;
  /** Durable bucket for output artifacts: checkpoints, policies, ONNX (RETAIN). */
  public readonly artifactBucket: s3.Bucket;
  /** ECR repo for the IL fine-tune container (the generalized `vla-ft` image). */
  public readonly vlaFtRepo: ecr.Repository;
  /** ECR repo for the RL training container (Isaac Lab headless PPO). */
  public readonly isaacLabRlRepo: ecr.Repository;
  /** ECR repo for the GR00T N1.7 fine-tune container (Isaac-GR00T, Cosmos backbone). */
  public readonly grootRepo: ecr.Repository;
  /**
   * Base managed policy granting a training job read on dataBucket, read/write on
   * artifactBucket, and pull on both ECR repos. Pattern stacks attach this to their
   * execution/job roles, then layer service-specific permissions on top.
   */
  public readonly jobBasePolicy: iam.ManagedPolicy;

  constructor(scope: Construct, id: string, props: SharedBaseStackProps = {}) {
    super(scope, id, props);

    const prefix = props.namePrefix ?? 'pai';

    // --- Network ---
    // 2 AZ for capacity resilience (AzSelector picks the live one at deploy time);
    // private-with-egress for GPU instances, public only for the NAT path.
    this.vpc = new ec2.Vpc(this, 'Vpc', {
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        {
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: 'private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
      ],
    });
    // Gateway endpoint keeps S3 traffic off the NAT (cost + throughput for datasets).
    this.vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    // --- Shared filesystem ---
    // Bursting throughput, encrypted; mounted by jobs that need cross-job scratch or
    // a checkpoint cache. RETAIN so an in-flight training run is never destroyed by a
    // stack teardown.
    this.fileSystem = new efs.FileSystem(this, 'SharedEfs', {
      vpc: this.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      encrypted: true,
      performanceMode: efs.PerformanceMode.GENERAL_PURPOSE,
      throughputMode: efs.ThroughputMode.BURSTING,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // --- Durable buckets (RETAIN: datasets + trained artifacts must survive teardown) ---
    const bucketCommon = {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      versioned: true,
    } as const;

    // Dedicated S3 server access-log destination. The data/artifact buckets log here
    // under distinct prefixes; this bucket is intentionally NOT self-logged (AWS warns
    // self-logging creates a recursive logs-about-logs loop), so it is the one bucket
    // for which access logging stays off by design.
    const accessLogsBucket = new s3.Bucket(this, 'AccessLogsBucket', {
      bucketName: `${prefix}-access-logs-${this.account}-${this.region}`,
      ...bucketCommon,
    });

    this.dataBucket = new s3.Bucket(this, 'DataBucket', {
      bucketName: `${prefix}-data-${this.account}-${this.region}`,
      ...bucketCommon,
      serverAccessLogsBucket: accessLogsBucket,
      serverAccessLogsPrefix: 'data/',
    });
    this.artifactBucket = new s3.Bucket(this, 'ArtifactBucket', {
      bucketName: `${prefix}-artifacts-${this.account}-${this.region}`,
      ...bucketCommon,
      serverAccessLogsBucket: accessLogsBucket,
      serverAccessLogsPrefix: 'artifacts/',
    });

    // --- Container registries ---
    // RETAIN repos; expire untagged images so the registry does not grow unbounded
    // across rebuilds, while keeping all tagged (released) images.
    const repoProps = (repositoryName: string): ecr.RepositoryProps => ({
      repositoryName,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      imageScanOnPush: true,
      // KMS at-rest encryption (AWS-managed key, no standing key cost) rather than the
      // ECR default of AES256.
      encryption: ecr.RepositoryEncryption.KMS,
      lifecycleRules: [
        {
          description: 'Expire untagged images after 14 days',
          tagStatus: ecr.TagStatus.UNTAGGED,
          maxImageAge: cdk.Duration.days(14),
        },
      ],
    });

    this.vlaFtRepo = new ecr.Repository(this, 'VlaFtRepo', repoProps(`${prefix}/vla-ft`));
    this.isaacLabRlRepo = new ecr.Repository(
      this,
      'IsaacLabRlRepo',
      repoProps(`${prefix}/isaac-lab-rl`),
    );
    this.grootRepo = new ecr.Repository(this, 'GrootRepo', repoProps(`${prefix}/gr00t-n17`));

    // --- Base job policy ---
    // Least-privilege scaffold pattern stacks attach to job/execution roles.
    this.jobBasePolicy = new iam.ManagedPolicy(this, 'JobBasePolicy', {
      description:
        'Base access for PAI training jobs: read dataBucket, read/write artifactBucket, pull ECR repos',
      statements: [
        new iam.PolicyStatement({
          sid: 'ReadDataBucket',
          actions: ['s3:GetObject', 's3:ListBucket'],
          resources: [this.dataBucket.bucketArn, this.dataBucket.arnForObjects('*')],
        }),
        new iam.PolicyStatement({
          sid: 'ReadWriteArtifactBucket',
          actions: ['s3:GetObject', 's3:PutObject', 's3:ListBucket', 's3:DeleteObject'],
          resources: [this.artifactBucket.bucketArn, this.artifactBucket.arnForObjects('*')],
        }),
        new iam.PolicyStatement({
          sid: 'PullContainers',
          actions: [
            'ecr:GetDownloadUrlForLayer',
            'ecr:BatchGetImage',
            'ecr:BatchCheckLayerAvailability',
          ],
          resources: [
            this.vlaFtRepo.repositoryArn,
            this.isaacLabRlRepo.repositoryArn,
            this.grootRepo.repositoryArn,
          ],
        }),
        new iam.PolicyStatement({
          // GetAuthorizationToken is not resource-scopable.
          sid: 'EcrAuth',
          actions: ['ecr:GetAuthorizationToken'],
          resources: ['*'],
        }),
      ],
    });

    // --- Outputs (cross-stack wiring + operator convenience) ---
    new cdk.CfnOutput(this, 'VpcId', { value: this.vpc.vpcId });
    new cdk.CfnOutput(this, 'EfsId', { value: this.fileSystem.fileSystemId });
    new cdk.CfnOutput(this, 'DataBucketName', { value: this.dataBucket.bucketName });
    new cdk.CfnOutput(this, 'ArtifactBucketName', { value: this.artifactBucket.bucketName });
    new cdk.CfnOutput(this, 'VlaFtRepoUri', { value: this.vlaFtRepo.repositoryUri });
    new cdk.CfnOutput(this, 'IsaacLabRlRepoUri', { value: this.isaacLabRlRepo.repositoryUri });
    new cdk.CfnOutput(this, 'GrootRepoUri', { value: this.grootRepo.repositoryUri });
  }
}
