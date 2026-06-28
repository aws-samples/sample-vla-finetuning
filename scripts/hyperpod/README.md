# Pattern C deploy runbook — HyperPod Slurm multi-node FSDP2

Pattern C (`lib/il/hyperpod-stack.ts`) is the >1-node IL tier: a standing SageMaker HyperPod
Slurm cluster running multi-node FSDP2 over EFA, with optional FSx Lustre + S3 DRA as the
shared data plane. It is **deploy-gated** (`-c enableHyperPod=true`) because the cluster is a
continuous multi-GPU cost — `2 × ml.g6e.48xlarge ≈ $75/hr` plus the head node, billing from
cluster-create until you tear it down (NOT per-job like the Batch CEs, which are $0 idle).

This directory holds the one piece the cluster cannot self-bootstrap: the **lifecycle bundle**
that HyperPod runs on every node at create. `cdk synth` proves the template; it does NOT stage
this bundle. Staging it is a prerequisite that must happen **before** `cdk deploy`, or every
node's provisioning fails and the cluster never forms Slurm.

## Prerequisites (one-time / per-account)

- `ml.g6e.48xlarge for cluster usage` quota ≥ 2 (L-177AF1AD). Request + approve in your account before deploy.
- `ml.c5.2xlarge for cluster usage` quota ≥ 1 (the Slurm controller) — broadly available (was 30).
- The EFA fabric image built + pushed: `build.sh --efa` → `pai/vla-ft:efa` in ECR (done).
- The `sagemaker-<region>-<account>` bucket exists (S3 auto-creates on first sync; HyperPod's
  managed role only grants the `sagemaker-` prefix, which is why the bundle lives there).

## Why a controller group exists

A HyperPod **Slurm** cluster needs a dedicated controller (head) instance group separate from
the GPU workers — `lifecycle_script.py` looks up `controller_group` by name and a single group
cannot be both head and compute. A worker-only cluster (the pre-2026-06-27 default) would synth
fine but never form Slurm at deploy. So the IL/RL stacks default to:

| group              | type            | count | role                     |
|--------------------|-----------------|-------|--------------------------|
| `controller-machine` | ml.c5.2xlarge | 1     | Slurm controller (head)  |
| `worker-group-1`     | ml.g6e.48xlarge | 2   | GPU compute (8×L40S each) |

The group **names** are declared in the CDK stack (`lib/il/hyperpod-stack.ts`) and carried into
`SlurmConfig.NodeType` per group. In managed-Slurm mode HyperPod generates
`provisioning_parameters.json` from that declaration — `stage-lifecycle.sh` no longer writes one.

## FSx (read before `-c hyperPodFsx=true`)

The recommended first run is **FSx-free** (validated 2026-06-28). Skip `-c hyperPodFsx=true`,
deploy, and have the launch job stage data to the workers' local NVMe (g6e.48xlarge has 4×1.9 TB).
This still validates the *expensive, risky* part — EFA fabric + 2-node NCCL all-reduce — with no
FSx complication.

For an FSx run, note managed-Slurm owns `provisioning_parameters.json`, so the legacy
`--fsx-id` regeneration flow does not apply; mount FSx via the cluster's FSx wiring
(`-c hyperPodFsx=true`) and the staged `mount_fsx.sh`, or add the FSx coordinates through the
managed-Slurm path. (`--fsx-id` is accepted by the script for back-compat but is a no-op here.)

## Step-by-step (FSx-free smoke — the recommended first deploy)

```bash
cd projects/vla-finetuning

# 1. Stage the verified lifecycle bundle (pinned awslabs commit, VERBATIM). In managed-Slurm
#    mode HyperPod generates provisioning_parameters.json itself — this script does NOT stage one.
scripts/hyperpod/stage-lifecycle.sh --region us-west-2 --dry-run
scripts/hyperpod/stage-lifecycle.sh --region us-west-2          # real sync

# 2. Deploy the cluster (gated). $/hr starts here. ALWAYS pass --exclusively so CDK acts on
#    ONLY this stack and never tries to UPDATE the shared Base stack (which has drifted and
#    would attempt a destructive ECR-repo replace — see "Base stack landmine" below).
npx cdk deploy PaiTrainingPlatform-IL-HyperPod --exclusively \
  -c enableHyperPod=true -c region=us-west-2 --require-approval never

# 3. Connect to the controller over SSM and confirm Slurm sees both worker nodes:
#    aws ssm start-session --target sagemaker-cluster:<cluster-id>_controller-machine-<instance-id>
#    sinfo  →  expect worker-group-1's 2 nodes in 'idle' state in the 'dev' partition.
#    (cluster-id from the ClusterArn stack output; instance-id from `aws sagemaker list-cluster-nodes`.)

# 4. Submit the 2-node EFA + NCCL smoke. With no shared FS, the job pulls its script from S3 to
#    each node's local disk and runs the :efa image over the host network. The minimal high-value
#    proof is a 2-node all-reduce (validates EFA RDMA + the :efa fabric); see the sbatch pattern
#    captured in the session notes. The full FSDP2 run (containers/vla-ft/hyperpod_fsdp_launch.sh)
#    assumes a shared /fsx and needs FSx-free local-staging adaptation first.
#    PASS = NCCL logs show NET/OFI (EFA), NOT NET/Socket, AND the all-reduce result is correct.

# 5. TEARDOWN — the cluster bills until destroyed. Do this as soon as the smoke is captured:
npx cdk destroy PaiTrainingPlatform-IL-HyperPod --exclusively \
  -c enableHyperPod=true -c region=us-west-2 --force
```

## Validated 1-shot deploy gaps (surfaced by the real 2026-06-28 deploy, all fixed in code)

`cdk synth` + staging dry-run are necessary but NOT sufficient — a real deploy surfaced gaps no
synth could catch. All are now fixed in `lib/shared/hyperpod-cluster.ts` (+ regression tests):

1. **ClusterRole VpcConfig perms.** `AmazonSageMakerClusterInstanceRolePolicy` has zero ec2
   actions → CREATE fails "Unable to retrieve subnets". Fixed with the awslabs
   `AdditionToEnableVpcConfig` statements (DescribeSubnets/Vpcs + ENI mgmt) as an inline policy.
2. **SlurmConfig.NodeType.** A Slurm cluster needs exactly one group typed `Controller` and the
   workers `Compute` (CFN values `Controller|Login|Compute`) → else "no InstanceGroup with
   Controller node type". The optional `PartitionNames` is omitted (its charset rejects
   instance-type dots; the lifecycle bundle names partitions).
3. **Orchestrator vs provisioning_parameters.json.** `Orchestrator:{Slurm}` = managed-Slurm mode,
   where HyperPod writes its own provisioning_parameters.json. Staging our own alongside it →
   "cannot include both a SLURM Orchestrator Config and a provisioning_parameters.json". So
   `stage-lifecycle.sh` no longer generates that file.
4. **EFA self-referencing SG egress.** EFA RDMA needs an all-traffic egress rule targeting the SG
   BY REFERENCE (the 0.0.0.0/0 egress from allowAllOutbound does NOT satisfy it). Without it, NCCL
   bootstrap (TCP) works and selects NET/OFI, but the cross-node all-reduce fails "Unreachable
   remote". Emitted at L1 (`CfnSecurityGroupEgress`; the L2 form is dropped under allowAllOutbound).

### Runtime gotcha (managed-Slurm + base-config) — automated (code), pending live re-validation

After the cluster reaches InService, the awslabs base-config bundle's accounting + topology setup
collides with HyperPod's managed slurm.conf, leaving `slurmctld` crash-looping:
`No Assoc usage file to recover` / `GresTypes specified more than once` / `No switches configured`.
The 2026-06-28 smoke unblocked it manually on the controller: comment `Include accounting.conf`
and `TopologyPlugin=topology/tree` in `/opt/slurm/etc/slurm.conf`, then `systemctl restart
slurmctld` → `sinfo` shows both workers idle.

That manual procedure is now **automated** by `fix_managed_slurm_conf.sh`, which `stage-lifecycle.sh`
copies into the bundle and wires into `on_create.sh`'s tail (one guarded call inserted before the
final `exit $exit_code`, so it runs *after* `lifecycle_script.py` → `start_slurm.sh` has started
slurmctld — exactly when the collision manifests). The fix is controller-only (workers rename
`slurmctld.service` away, which the script detects), idempotent (a `# pai-managed-slurm-reconciled`
sentinel marker makes re-runs no-ops), and never-fatal (returns 0 in every path; `on_create.sh`
runs under `set -e`). The **pinned base-config files stay byte-for-byte verbatim** — we add our own
script and patch only the *staged copy* of `on_create.sh`, never the pinned source tree (same
belt-and-suspenders pattern as the `provisioning_parameters.json` removal).

> ⚠️ Validated at the staging/injection layer (bash -n, isolated patch-and-parse of the real
> pinned `on_create.sh`), and it replicates the procedure that worked on the live 2026-06-28
> cluster — but it has **not yet been re-validated end-to-end on a fresh $75/hr deploy** (the
> collision only manifests on a real controller at InService). The next Pattern C deploy should
> confirm `sinfo` shows both workers idle WITHOUT the manual step. If it does, drop this caveat.

## Base stack landmine (`--exclusively` is mandatory)

`cdk deploy <stack>` auto-includes dependency stacks. The deployed `PaiTrainingPlatform-Base`
has drifted from the local code (it wants to add an AccessLogsBucket and **replace** the 3 ECR
repos — which would destroy the `:efa` image). CFN refuses the custom-named replace and rolls
Base back, but it's a wasted cycle and a real risk. `--exclusively` makes CDK act on ONLY the
HyperPod stack; the VPC subnet imports resolve from the live Base exports regardless.

## Files

- `stage-lifecycle.sh` — fetch the verified base-config bundle VERBATIM from the pinned commit
  (`34a09f12…`, MIT-0) and sync it to the cluster's `OnCreate` S3 prefix. Does NOT generate
  provisioning_parameters.json (managed-Slurm mode — HyperPod does). It also copies
  `fix_managed_slurm_conf.sh` into the bundle and wires it into the staged `on_create.sh` tail
  (see the runtime gotcha above). `--dry-run` to preview. Bump the `PIN` deliberately, never
  float to `main`.
- `fix_managed_slurm_conf.sh` — OUR additive controller-side reconciliation of the managed
  slurm.conf vs the verbatim base-config (comments the colliding `Include accounting.conf` +
  `TopologyPlugin` directives, restarts slurmctld). Idempotent, controller-only, never-fatal.
  Staged into the bundle and invoked from `on_create.sh` by `stage-lifecycle.sh`.
- The launch script lives with the container: `containers/vla-ft/hyperpod_fsdp_launch.sh`.

## Verification status

**VALIDATED end-to-end (2026-06-28):** real 2-node deploy → cluster InService → Slurm formed
(2 workers idle) → 2-node × 8-GPU = 16-rank NCCL all-reduce over EFA RDMA (world=16, OK=True) →
torn down. ~1 hr of g6e.48xlarge×2 + controller (~$75). The 4 code gaps above are fixed and
locked by tests; the runtime slurm.conf gotcha still needs a manual step (or a durable fix).
