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
  /**
   * Cluster instance groups. Default: a Slurm controller (`controller-machine`,
   * ml.c5.2xlarge ×1) PLUS 2× ml.g6e.48xlarge workers (`worker-group-1`, 8×L40S each =
   * 16 GPU). A HyperPod *Slurm* cluster REQUIRES a dedicated controller group separate
   * from the workers (the lifecycle_script.py looks up `controller_group` by name and a
   * single group cannot be both head + compute) — a worker-only cluster never forms Slurm.
   * The group names here MUST match the `controller_group` / `worker_groups[].instance_group_name`
   * in the staged provisioning_parameters.json (see scripts/hyperpod/stage-lifecycle.sh).
   * An operator overriding this MUST keep a controller group.
   */
  readonly instanceGroups?: HyperPodInstanceGroup[];
  /**
   * S3 URI (must start with s3://sagemaker-) of the openvla provisioning lifecycle
   * scripts. Default points at a conventional path in the artifact bucket's sagemaker-
   * prefixed sibling; the operator stages real scripts there before deploy.
   */
  readonly lifecycleS3Uri?: string;
  /**
   * Attach an FSx for Lustre + S3 DRA hot tier (the multi-node data plane, Phase 2). When
   * true, the cluster mounts an FSx Lustre filesystem linked to the platform dataBucket —
   * datasets + DCP checkpoints hydrate from S3, shared across all ranks over EFA. Default
   * false (synth check / no shared FS). The FS bills continuously, so enable only for a
   * gated multi-node deploy. Wire via `-c enableHyperPod=true -c hyperPodFsx=true`.
   */
  readonly attachFsx?: boolean;
  /** FSx Lustre capacity in GiB when attachFsx (PERSISTENT_2 valid values). Default 9600. */
  readonly fsxStorageCapacityGiB?: number;
}

export class IlHyperPodStack extends cdk.Stack {
  public readonly hyperPod: HyperPodCluster;

  constructor(scope: Construct, id: string, props: IlHyperPodStackProps) {
    super(scope, id, props);

    const namePrefix = props.namePrefix ?? 'pai';
    // A Slurm HyperPod cluster needs a dedicated controller (head) group + worker group(s).
    // Controller is a small CPU box (ml.c5.2xlarge — cluster quota is broadly available,
    // unlike the g-series); workers are the GPU compute. The group names are the verified
    // awslabs convention (`controller-machine` / `worker-group-1`) and MUST match the staged
    // provisioning_parameters.json controller_group / worker_groups[].instance_group_name.
    const instanceGroups = props.instanceGroups ?? [
      { name: 'controller-machine', instanceType: 'ml.c5.2xlarge', instanceCount: 1, ebsGb: 100, nodeType: 'Controller' as const },
      { name: 'worker-group-1', instanceType: 'ml.g6e.48xlarge', instanceCount: 2, ebsGb: 500, nodeType: 'Compute' as const },
    ];

    this.hyperPod = new HyperPodCluster(this, 'IlCluster', {
      base: props.base,
      clusterName: `${namePrefix}-il-hyperpod`,
      instanceGroups,
      // HyperPod requires the s3://sagemaker- prefix for lifecycle scripts.
      lifecycleS3Uri:
        props.lifecycleS3Uri ?? `s3://sagemaker-${this.region}-${this.account}/${namePrefix}/il-hyperpod-lifecycle/`,
      // Multi-node data plane: mount an FSx Lustre hot tier linked to the platform
      // dataBucket (datasets in, DCP checkpoints out) when attachFsx.
      ...(props.attachFsx
        ? {
            fsxDataRepositoryBucket: props.base.dataBucket,
            fsxStorageCapacityGiB: props.fsxStorageCapacityGiB,
          }
        : {}),
    });
  }
}
