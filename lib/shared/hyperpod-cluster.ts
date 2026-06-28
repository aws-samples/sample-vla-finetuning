/**
 * HyperPodCluster — shared construct for Pattern C (SageMaker HyperPod) clusters.
 *
 * Pattern C is the multi-node tier of the decision rule table (ARCHITECTURE §3.1:
 * "exceeds single node → HyperPod"). Both axes use it the same way — a persistent,
 * resilient GPU cluster for distributed training that out-survives a single Training
 * Job / Batch job — so the cluster wiring is ONE construct and the IL and RL stacks
 * differ only in which container/lifecycle scripts run on it and the instance-group
 * sizing.
 *
 * Copy-paste starting point per the asset map: `aws-samples/awsome-distributed-training`
 * openvla (Slurm) / openvla-oft (EKS). We model the **Slurm** orchestrator: it is HyperPod's canonical
 * default and is self-contained at synth time, whereas the EKS orchestrator's config
 * requires the ARN of a pre-existing EKS cluster (an external dependency that would
 * make `cdk synth` non-self-contained). Swapping to EKS later is a localized change to
 * the `orchestrator` block.
 *
 * VERIFICATION ASYMMETRY (user-set, ARCHITECTURE §6): Pattern C is **code + `cdk synth`
 * only** — a real multi-node deploy is deferred (a HyperPod cluster is a multi-hour,
 * standing-cost commitment, and g-series *cluster* quotas are 0 even though g6e
 * *training-job* quota auto-approved — see ROADMAP open question). This construct
 * therefore synthesizes a complete, deployable template but is wired into NO stack
 * that deploys by default. When a multi-node job justifies the cluster, an operator
 * sets the instance-group count and the lifecycle S3 URI and deploys.
 *
 * Standing cost when deployed is high (N × multi-GPU instances run continuously until
 * the cluster is torn down), which is exactly why it is deferred and gated behind an
 * explicit instance count.
 */
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import { Construct } from 'constructs';
import { SharedBaseStack } from './base-stack';
import { FsxLustreFileSystem } from './fsx-lustre';

export interface HyperPodInstanceGroup {
  /** Instance-group name (e.g. 'worker', 'controller'). */
  readonly name: string;
  /** Instance type, e.g. 'ml.g6e.12xlarge' (4×L40S) or 'ml.p5.48xlarge' (8×H100). */
  readonly instanceType: string;
  /** Number of instances in this group. The deploy gate: 0 elsewhere keeps cost off. */
  readonly instanceCount: number;
  /** Per-instance EBS GB. Default 500 (image + dataset + checkpoints staging). */
  readonly ebsGb?: number;
  /**
   * Slurm node type for this group (CfnCluster SlurmConfig). A HyperPod Slurm cluster
   * REQUIRES exactly one group with nodeType 'Controller'; the GPU workers are 'Compute'.
   * CloudFormation rejects CREATE ("no InstanceGroup with Controller node type") if no
   * group declares it. Allowed values: 'Controller' | 'Login' | 'Compute'. Default
   * 'Compute' (workers are the common case) — the head group must set 'Controller'.
   */
  readonly nodeType?: 'Controller' | 'Login' | 'Compute';
}

export interface HyperPodClusterProps {
  /** Shared base — VPC + jobBasePolicy (S3/ECR) are reused for the cluster. */
  readonly base: SharedBaseStack;
  /** Cluster name (also the physical CfnCluster name). */
  readonly clusterName: string;
  /** One or more instance groups (controller / worker tiers). */
  readonly instanceGroups: HyperPodInstanceGroup[];
  /**
   * S3 URI of the HyperPod lifecycle scripts (provisioning bootstrap run on each
   * node at create). HyperPod REQUIRES this to start with `s3://sagemaker-` (the
   * managed AmazonSageMakerClusterInstanceRolePolicy only grants that prefix). The
   * operator stages the openvla/openvla-oft provisioning scripts here before deploy.
   */
  readonly lifecycleS3Uri: string;
  /** Entrypoint script filename under lifecycleS3Uri. Default 'on_create.sh'. */
  readonly lifecycleOnCreate?: string;
  /**
   * Attach an FSx for Lustre + S3 DRA hot tier the cluster nodes mount (the multi-node
   * data plane — datasets + DCP checkpoints hydrate from S3, shared across all ranks over
   * EFA). When set, an FsxLustreFileSystem is created in the platform VPC and linked to this
   * bucket; the cluster SG is allowed to reach the FSx SG. Omit (default) for a cluster
   * with no shared FS (e.g. a synth check, or a job that stages to local NVMe). The FSx FS
   * bills continuously while it exists — only set this for a genuinely gated multi-node deploy.
   */
  readonly fsxDataRepositoryBucket?: cdk.aws_s3.IBucket;
  /** S3 prefix under fsxDataRepositoryBucket the DRA links. Default '' (whole bucket). */
  readonly fsxDataRepositoryPrefix?: string;
  /** FSx Lustre capacity in GiB (PERSISTENT_2: 1200/2400/multiples of 2400). Default 9600. */
  readonly fsxStorageCapacityGiB?: number;
}

export class HyperPodCluster extends Construct {
  public readonly cluster: sagemaker.CfnCluster;
  /** Cluster instance role (the nodes' identity: S3/ECR + HyperPod managed policy). */
  public readonly clusterRole: iam.Role;
  public readonly securityGroup: ec2.SecurityGroup;
  /** The attached FSx Lustre hot tier, when fsxDataRepositoryBucket was supplied. */
  public readonly fsx?: FsxLustreFileSystem;

  constructor(scope: Construct, id: string, props: HyperPodClusterProps) {
    super(scope, id);

    const { base } = props;

    // Cluster nodes need: the HyperPod managed policy (cluster lifecycle, the
    // `sagemaker-` S3 prefix for lifecycle scripts) + our jobBasePolicy (dataset
    // read, artifact write, ECR pull of the training image).
    //
    // PLUS the VpcConfig permissions. AmazonSageMakerClusterInstanceRolePolicy grants
    // logs/cloudwatch/s3(sagemaker-)/ssm but NO ec2 actions, so a VPC-configured cluster
    // fails CREATE at validation with "Unable to retrieve subnets" (the execution role
    // can't DescribeSubnets/DescribeVpcs). The canonical awslabs HyperPod reference ships
    // these as a separate execution-role policy; we attach the two ec2 statements VERBATIM
    // from the pinned source (1.AmazonSageMakerClustersExecutionRolePolicy.json @ 34a09f12,
    // Sids AdditionToEnableVpcConfig / Addition2ToEnableVpcConfig) — a match-verified lock,
    // not a hand-assembled permission set.
    this.clusterRole = new iam.Role(this, 'ClusterRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      description: `HyperPod cluster instance role for ${props.clusterName}`,
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerClusterInstanceRolePolicy'),
        base.jobBasePolicy,
      ],
      inlinePolicies: {
        EnableVpcConfig: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              sid: 'AdditionToEnableVpcConfig',
              effect: iam.Effect.ALLOW,
              actions: [
                'ec2:CreateNetworkInterface',
                'ec2:CreateNetworkInterfacePermission',
                'ec2:DeleteNetworkInterface',
                'ec2:DeleteNetworkInterfacePermission',
                'ec2:DescribeNetworkInterfaces',
                'ec2:DescribeVpcs',
                'ec2:DescribeDhcpOptions',
                'ec2:DescribeSubnets',
                'ec2:DescribeSecurityGroups',
                'ec2:DetachNetworkInterface',
              ],
              resources: ['*'],
            }),
            new iam.PolicyStatement({
              sid: 'Addition2ToEnableVpcConfig',
              effect: iam.Effect.ALLOW,
              actions: ['ec2:CreateTags'],
              resources: ['arn:aws:ec2:*:*:network-interface/*'],
            }),
          ],
        }),
      },
    });

    // Cluster nodes communicate over NCCL (distributed training). Keep allowAllOutbound:true
    // for the 0.0.0.0/0 egress that ECR image pulls + S3 (dataset/lifecycle) need over NAT.
    this.securityGroup = new ec2.SecurityGroup(this, 'ClusterSg', {
      vpc: base.vpc,
      allowAllOutbound: true,
      description: `HyperPod ${props.clusterName} inter-node (NCCL/EFA) + egress`,
    });
    // Ingress: all traffic from self (the NCCL/rendezvous + EFA data plane source side).
    this.securityGroup.connections.allowInternally(
      ec2.Port.allTraffic(),
      'HyperPod inter-node NCCL / EFA RDMA (ingress self-ref)',
    );
    // EFA RDMA ALSO requires an all-traffic egress rule that targets the SG BY REFERENCE
    // (not the 0.0.0.0/0 CIDR allowAllOutbound gives). Without it, NCCL bootstrap over TCP
    // succeeds but the EFA data plane fails the cross-node all-reduce with "Unreachable
    // remote (never received a response)". The L2 connections.allowTo(self) is SILENTLY
    // DROPPED when allowAllOutbound:true, so emit the rule at L1 (CfnSecurityGroupEgress).
    // Verified by a 2-node 16-rank NCCL all-reduce (world=16, OK=True).
    new ec2.CfnSecurityGroupEgress(this, 'ClusterSgEfaSelfEgress', {
      groupId: this.securityGroup.securityGroupId,
      ipProtocol: '-1',
      destinationSecurityGroupId: this.securityGroup.securityGroupId,
      description: 'HyperPod inter-node EFA RDMA (egress self-ref — required by EFA)',
    });

    const subnetIds = base.vpc.selectSubnets({
      subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
    }).subnetIds;

    const lifeCycleConfig: sagemaker.CfnCluster.ClusterLifeCycleConfigProperty = {
      sourceS3Uri: props.lifecycleS3Uri,
      onCreate: props.lifecycleOnCreate ?? 'on_create.sh',
    };

    this.cluster = new sagemaker.CfnCluster(this, 'Cluster', {
      clusterName: props.clusterName,
      // Slurm orchestrator: self-contained at synth (EKS needs an external cluster ARN).
      orchestrator: { slurm: {} },
      instanceGroups: props.instanceGroups.map((g) => ({
        instanceGroupName: g.name,
        instanceType: g.instanceType,
        instanceCount: g.instanceCount,
        executionRole: this.clusterRole.roleArn,
        lifeCycleConfig,
        threadsPerCore: 1,
        instanceStorageConfigs: [
          { ebsVolumeConfig: { volumeSizeInGb: g.ebsGb ?? 500 } },
        ],
        // Each group declares only its Slurm role. HyperPod CREATE fails ("no InstanceGroup
        // with Controller node type") unless exactly one group is 'Controller'. We omit the
        // optional CfnCluster PartitionNames: the verified awslabs flow names Slurm partitions
        // from the staged provisioning_parameters.json (worker_groups[].partition_name), and
        // the CfnCluster PartitionNames charset (^[a-zA-Z0-9](-*[a-zA-Z0-9])*$) rejects the
        // instance-type dots anyway. Partition setup stays in the lifecycle bundle, by design.
        slurmConfig: { nodeType: g.nodeType ?? 'Compute' },
      })),
      vpcConfig: {
        securityGroupIds: [this.securityGroup.securityGroupId],
        subnets: subnetIds,
      },
      // Auto-replace a failed node (resilience is HyperPod's whole point vs Batch/SM).
      nodeRecovery: 'Automatic',
    });

    // --- Optional FSx Lustre hot tier (the multi-node data plane, Phase 2 construct) ---
    // When a data-repository bucket is supplied, create an FSx Lustre + S3 DRA the cluster
    // nodes mount (shared dataset + DCP checkpoint tier, hydrated from S3 over EFA). The
    // cluster SG is allowed into the FSx SG so nodes can reach the Lustre client ports.
    if (props.fsxDataRepositoryBucket) {
      this.fsx = new FsxLustreFileSystem(this, 'Fsx', {
        vpc: base.vpc,
        dataRepositoryBucket: props.fsxDataRepositoryBucket,
        dataRepositoryPrefix: props.fsxDataRepositoryPrefix,
        storageCapacityGiB: props.fsxStorageCapacityGiB,
        // Pin FSx to the same private subnet the cluster lands in (Lustre is single-AZ;
        // co-locating with the nodes avoids cross-AZ data-transfer cost + latency).
        vpcSubnet: base.vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }).subnets[0],
      });
      // Let the cluster nodes reach the FSx filesystem (Lustre client ports live in the
      // FSx SG's self-referencing all-traffic rule; add the cluster SG as an allowed source).
      this.fsx.securityGroup.connections.allowFrom(
        this.securityGroup,
        ec2.Port.allTraffic(),
        'HyperPod cluster nodes mount the FSx Lustre hot tier',
      );
      new cdk.CfnOutput(this, 'FsxFileSystemId', {
        value: this.fsx.fileSystem.fileSystemId,
        description: 'FSx Lustre filesystem id the cluster mounts (multi-node data plane)',
      });
      new cdk.CfnOutput(this, 'FsxMountName', {
        value: this.fsx.fileSystem.mountName,
        description: 'FSx Lustre mount name (lifecycle on_create.sh mounts this)',
      });
      new cdk.CfnOutput(this, 'FsxDataRepositoryPath', {
        value: this.fsx.dataRepositoryPath,
        description: 'S3 path the DRA links (bidirectional NEW/CHANGED/DELETED sync)',
      });
    }

    new cdk.CfnOutput(this, 'ClusterName', {
      value: props.clusterName,
      description: 'HyperPod cluster name (deploy is gated; see stack docs)',
    });
    new cdk.CfnOutput(this, 'ClusterArn', {
      value: this.cluster.attrClusterArn,
      description: 'HyperPod cluster ARN',
    });
  }
}
