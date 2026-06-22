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

export interface HyperPodInstanceGroup {
  /** Instance-group name (e.g. 'worker', 'controller'). */
  readonly name: string;
  /** Instance type, e.g. 'ml.g6e.12xlarge' (4×L40S) or 'ml.p5.48xlarge' (8×H100). */
  readonly instanceType: string;
  /** Number of instances in this group. The deploy gate: 0 elsewhere keeps cost off. */
  readonly instanceCount: number;
  /** Per-instance EBS GB. Default 500 (image + dataset + checkpoints staging). */
  readonly ebsGb?: number;
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
}

export class HyperPodCluster extends Construct {
  public readonly cluster: sagemaker.CfnCluster;
  /** Cluster instance role (the nodes' identity: S3/ECR + HyperPod managed policy). */
  public readonly clusterRole: iam.Role;
  public readonly securityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: HyperPodClusterProps) {
    super(scope, id);

    const { base } = props;

    // Cluster nodes need: the HyperPod managed policy (cluster lifecycle, the
    // `sagemaker-` S3 prefix for lifecycle scripts) + our jobBasePolicy (dataset
    // read, artifact write, ECR pull of the training image).
    this.clusterRole = new iam.Role(this, 'ClusterRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      description: `HyperPod cluster instance role for ${props.clusterName}`,
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerClusterInstanceRolePolicy'),
        base.jobBasePolicy,
      ],
    });

    // Cluster nodes communicate over NCCL (distributed training) — allow all traffic
    // within the SG (self-referencing) plus egress for ECR/S3 over NAT.
    this.securityGroup = new ec2.SecurityGroup(this, 'ClusterSg', {
      vpc: base.vpc,
      allowAllOutbound: true,
      description: `HyperPod ${props.clusterName} inter-node (NCCL) + egress`,
    });
    this.securityGroup.connections.allowInternally(
      ec2.Port.allTraffic(),
      'HyperPod inter-node NCCL / distributed rendezvous',
    );

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
      })),
      vpcConfig: {
        securityGroupIds: [this.securityGroup.securityGroupId],
        subnets: subnetIds,
      },
      // Auto-replace a failed node (resilience is HyperPod's whole point vs Batch/SM).
      nodeRecovery: 'Automatic',
    });

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
