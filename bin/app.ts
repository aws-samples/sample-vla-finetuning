#!/usr/bin/env node
/**
 * CDK App entrypoint for the PAI Training Platform.
 *
 * Composes the shared base stack (network, EFS, ECR, S3, base IAM) plus the IL
 * Pattern B stack (SageMaker Training Job execution role). Other pattern stacks
 * (A Batch, RL Batch, HyperPod) are added in later phases and import the base
 * stack's resources rather than re-creating them.
 *
 * Configuration is by CDK context (`-c key=value`) and the standard CDK env vars —
 * nothing account-specific is hardcoded. The AWS account is taken from
 * CDK_DEFAULT_ACCOUNT (your active credentials), the region from `-c region=` or
 * CDK_DEFAULT_REGION, and the platform's own buckets/repos derive their names from
 * `${namePrefix}-...-${account}-${region}` at deploy time.
 *
 *   cdk synth                         # synth all stacks
 *   cdk deploy --all -c region=us-west-2          # deploy base + Pattern B
 *   cdk deploy PaiTrainingPlatform-Base           # deploy just the base
 *   cdk deploy -c namePrefix=pai-dev  # isolate a second deployment in one account
 *
 * Optional: point the IL job roles at your OWN already-populated LeRobot dataset
 * buckets (read-only), so a fine-tune can read them without first copying data into
 * the platform dataBucket:
 *   cdk deploy --all -c ilDatasetBucketArn=arn:aws:s3:::my-il-dataset-bucket
 *   cdk deploy --all -c grootDatasetBucketArn=arn:aws:s3:::my-groot-dataset-bucket
 * Omit them and the patterns simply read the platform dataBucket (upload there first).
 */
import * as cdk from 'aws-cdk-lib';
import { SharedBaseStack } from '../lib/shared/base-stack';
import { PatternAStack } from '../lib/il/pattern-a-stack';
import { PatternBStack } from '../lib/il/pattern-b-stack';
import { GrootPatternAStack } from '../lib/il/gr00t-pattern-a-stack';
import { RlPatternAStack } from '../lib/rl/pattern-a-stack';

const app = new cdk.App();

const namePrefix = app.node.tryGetContext('namePrefix') ?? 'pai';
const region = app.node.tryGetContext('region') ?? process.env.CDK_DEFAULT_REGION;
// Opt-in email alerts on training-job completion/failure. Wire via
//   cdk deploy --all -c region=us-west-2 -c notifyEmail=you@example.com
const notifyEmail = app.node.tryGetContext('notifyEmail') as string | undefined;
// RL Pattern A capacity: Spot by default; `-c rlUseSpot=false` deploys an On-Demand CE
// for a reclaim-free run (first E2E smoke / short runs with no checkpoint to resume).
const rlUseSpot = app.node.tryGetContext('rlUseSpot') !== 'false';

// IL Pattern A capacity is per-JOB, not per-deploy: PatternAStack always deploys BOTH a
// Spot and an On-Demand queue (idle CEs cost nothing), and the launcher picks one at
// submit time via the plan's spot flag. So there is no ilUseSpot deploy toggle anymore —
// switching Spot↔On-Demand needs no redeploy.

// GR00T Pattern A capacity: On-Demand by default (this single long fine-tune has no
// EFS-resume, so a Spot reclaim would restart from scratch). `-c grootUseSpot=true` opts in.
const grootUseSpot = app.node.tryGetContext('grootUseSpot') === 'true';

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region,
};

const base = new SharedBaseStack(app, 'PaiTrainingPlatform-Base', {
  env,
  namePrefix,
  // The notification SNS topic is owned here (one per platform) so multiple pattern
  // stacks can route to it without colliding on the fixed topic name. Pattern stacks
  // import base.notificationTopic and add only their own EventBridge rule.
  notifyEmail,
  description:
    'PAI Training Platform — shared base (VPC, EFS, ECR, S3, base IAM) for IL + RL training backends',
});

// Optional pre-existing LeRobot dataset buckets the IL job roles may read directly
// (read-only), so a fine-tune can run without first copying ~GB of data into the
// platform dataBucket. Supply your own via `-c ilDatasetBucketArn=...` /
// `-c grootDatasetBucketArn=...`; omit them to read only the platform dataBucket.
const ilDatasetBucketArn = app.node.tryGetContext('ilDatasetBucketArn') as string | undefined;
const grootDatasetBucketArn = app.node.tryGetContext('grootDatasetBucketArn') as string | undefined;
const ilExtraDatasetArns = ilDatasetBucketArn ? [ilDatasetBucketArn] : [];
const grootExtraDatasetArns = grootDatasetBucketArn ? [grootDatasetBucketArn] : [];

// IL Pattern A (AWS Batch + g6e Spot): single-GPU tier for small/fast fine-tunes.
// Owns the Batch CE/JQ/JD; runs the same verified vla-ft container as Pattern B.
new PatternAStack(app, 'PaiTrainingPlatform-IL-PatternA', {
  env,
  namePrefix,
  base,
  extraDatasetReadArns: ilExtraDatasetArns,
  notifyEmail,
  description:
    'PAI Training Platform — IL Pattern A (AWS Batch + g6e for the vla-ft container)',
});

// IL Pattern B (SageMaker Training Job): execution role + launch wiring.
new PatternBStack(app, 'PaiTrainingPlatform-IL-PatternB', {
  env,
  namePrefix,
  base,
  extraDatasetReadArns: ilExtraDatasetArns,
  notifyEmail,
  description:
    'PAI Training Platform — IL Pattern B (SageMaker Training Job execution role for the vla-ft container)',
});

// IL GR00T Pattern A (AWS Batch + g6e On-Demand): GR00T N1.7 (3B Cosmos VLA) fine-tune.
// Third IL engine; owns its own Batch CE/JQ/JD running the gr00t-n17 container. g6e L40S
// only (A10G OOMs a 3B model), On-Demand (no EFS-resume → reclaim would restart).
new GrootPatternAStack(app, 'PaiTrainingPlatform-IL-GrootPatternA', {
  env,
  namePrefix,
  base,
  extraDatasetReadArns: grootExtraDatasetArns,
  notifyEmail,
  useSpot: grootUseSpot,
  description:
    'PAI Training Platform — IL GR00T Pattern A (AWS Batch + g6e for the GR00T N1.7 fine-tune container)',
});

// RL Pattern A (AWS Batch + g6e Spot): single-GPU tier for Isaac Lab headless PPO.
// Owns the Batch CE/JQ/JD (the CE/JQ/JD a verified Isaac Lab infra template left
// manual); runs the isaac-lab-rl container. Closes the Coverage gap (IL & RL, both real).
new RlPatternAStack(app, 'PaiTrainingPlatform-RL-PatternA', {
  env,
  namePrefix,
  base,
  notifyEmail,
  useSpot: rlUseSpot,
  description:
    'PAI Training Platform — RL Pattern A (AWS Batch + g6e for the Isaac Lab headless-PPO container)',
});
