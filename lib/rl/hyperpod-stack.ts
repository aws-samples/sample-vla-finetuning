/**
 * RlHyperPodStack — RL axis, Pattern C (SageMaker HyperPod).
 *
 * The multi-node tier for RL: distributed PPO across many nodes with NCCL, when a
 * single node's GPU count caps the parallel-env throughput. Copy-paste base:
 * `aws-samples/awsome-distributed-training` openvla-oft (EKS) and a verified Isaac Lab
 * Slurm multi-node pattern. Thin wrapper over the shared `HyperPodCluster`; the RL-specific
 * parts are the instance groups and (at deploy time) the Isaac Lab headless-PPO image
 * + provisioning scripts.
 *
 * Same verification asymmetry as the IL HyperPod stack: code + `cdk synth` ONLY, real
 * multi-node deploy deferred, NOT wired into bin/app.ts by default. The RL container
 * (isaac-lab-rl) and the verified reference task land in Phase 3; this stack is the
 * pre-built Pattern C target that Phase 3's RL job graduates to when it needs N nodes.
 */
import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { SharedBaseStack } from '../shared/base-stack';
import { HyperPodCluster, HyperPodInstanceGroup } from '../shared/hyperpod-cluster';

export interface RlHyperPodStackProps extends cdk.StackProps {
  readonly base: SharedBaseStack;
  readonly namePrefix?: string;
  /**
   * Cluster instance groups. Default: a Slurm controller (`controller-machine`,
   * ml.c5.2xlarge ×1) PLUS 2× ml.g6.12xlarge workers (`worker-group-1`, 4 GPU each) for
   * distributed PPO. A HyperPod Slurm cluster REQUIRES a dedicated controller group separate
   * from the workers — see IlHyperPodStack for the rationale. Names must match the staged
   * provisioning_parameters.json.
   */
  readonly instanceGroups?: HyperPodInstanceGroup[];
  /** S3 URI (s3://sagemaker-...) of the Isaac Lab RL provisioning lifecycle scripts. */
  readonly lifecycleS3Uri?: string;
}

export class RlHyperPodStack extends cdk.Stack {
  public readonly hyperPod: HyperPodCluster;

  constructor(scope: Construct, id: string, props: RlHyperPodStackProps) {
    super(scope, id, props);

    const namePrefix = props.namePrefix ?? 'pai';
    // Slurm controller (head) group + GPU worker group(s) — see IlHyperPodStack.
    const instanceGroups = props.instanceGroups ?? [
      { name: 'controller-machine', instanceType: 'ml.c5.2xlarge', instanceCount: 1, ebsGb: 100, nodeType: 'Controller' as const },
      { name: 'worker-group-1', instanceType: 'ml.g6.12xlarge', instanceCount: 2, ebsGb: 500, nodeType: 'Compute' as const },
    ];

    this.hyperPod = new HyperPodCluster(this, 'RlCluster', {
      base: props.base,
      clusterName: `${namePrefix}-rl-hyperpod`,
      instanceGroups,
      lifecycleS3Uri:
        props.lifecycleS3Uri ?? `s3://sagemaker-${this.region}-${this.account}/${namePrefix}/rl-hyperpod-lifecycle/`,
    });
  }
}
