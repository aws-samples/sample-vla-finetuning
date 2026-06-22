/**
 * IlHyperPodStack — IL axis, Pattern C (SageMaker HyperPod).
 *
 * The multi-node tier for IL when a fine-tune exceeds a single node (very large VLA
 * or a model/data scale that won't fit one g6e.48xlarge). Copy-paste base:
 * `aws-samples/awsome-distributed-training` openvla (Slurm). Thin wrapper over the shared
 * `HyperPodCluster` construct — the IL-specific parts are the instance-group sizing
 * and (at deploy time) the openvla provisioning lifecycle scripts + the vla-ft image.
 *
 * VERIFICATION = code + `cdk synth` ONLY (user-set asymmetry). Real multi-node deploy
 * deferred: a HyperPod cluster is a standing multi-GPU commitment and g-series cluster
 * quotas are 0. So this stack is NOT wired into bin/app.ts by default — it synthesizes
 * a complete template, ready to deploy when a job justifies the cluster. The default
 * instanceCount here is the *synth shape*; an operator sets the real count at deploy.
 */
import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { SharedBaseStack } from '../shared/base-stack';
import { HyperPodCluster, HyperPodInstanceGroup } from '../shared/hyperpod-cluster';

export interface IlHyperPodStackProps extends cdk.StackProps {
  readonly base: SharedBaseStack;
  readonly namePrefix?: string;
  /** Worker instance groups. Default: 2× ml.g6e.48xlarge (8×L40S each = 16 GPU). */
  readonly instanceGroups?: HyperPodInstanceGroup[];
  /**
   * S3 URI (must start with s3://sagemaker-) of the openvla provisioning lifecycle
   * scripts. Default points at a conventional path in the artifact bucket's sagemaker-
   * prefixed sibling; the operator stages real scripts there before deploy.
   */
  readonly lifecycleS3Uri?: string;
}

export class IlHyperPodStack extends cdk.Stack {
  public readonly hyperPod: HyperPodCluster;

  constructor(scope: Construct, id: string, props: IlHyperPodStackProps) {
    super(scope, id, props);

    const namePrefix = props.namePrefix ?? 'pai';
    const instanceGroups = props.instanceGroups ?? [
      { name: 'worker', instanceType: 'ml.g6e.48xlarge', instanceCount: 2, ebsGb: 500 },
    ];

    this.hyperPod = new HyperPodCluster(this, 'IlCluster', {
      base: props.base,
      clusterName: `${namePrefix}-il-hyperpod`,
      instanceGroups,
      // HyperPod requires the s3://sagemaker- prefix for lifecycle scripts.
      lifecycleS3Uri:
        props.lifecycleS3Uri ?? `s3://sagemaker-${this.region}-${this.account}/${namePrefix}/il-hyperpod-lifecycle/`,
    });
  }
}
