import * as cdk from 'aws-cdk-lib';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { SharedBaseStack } from '../lib/shared/base-stack';
import { RlPatternAStack } from '../lib/rl/pattern-a-stack';
import { stripPyCommentLines } from '../lib/shared/py-bootstrap';

const ENV = { account: '111111111111', region: 'us-west-2' };

describe('RlPatternAStack', () => {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'TestBase', { env: ENV, namePrefix: 'pai' });
  const stack = new RlPatternAStack(app, 'TestRlPatternA', {
    env: ENV,
    namePrefix: 'pai',
    base,
  });
  const t = Template.fromStack(stack);
  /** The job definition whose logical id starts with `prefix` (the stack has two). */
  const jobDef = (prefix: string): any => {
    const hits = Object.entries(t.findResources('AWS::Batch::JobDefinition')).filter(([id]) => id.startsWith(prefix));
    expect(hits).toHaveLength(1);
    return hits[0][1];
  };

  test('creates a managed EC2 Batch compute environment on Spot (g6e→g5 single-GPU fallback)', () => {
    t.resourceCountIs('AWS::Batch::ComputeEnvironment', 1);
    t.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      Type: 'managed',
      ComputeResources: Match.objectLike({
        Type: 'SPOT',
        InstanceTypes: ['g6e.4xlarge', 'g5.4xlarge'],
        AllocationStrategy: 'SPOT_PRICE_CAPACITY_OPTIMIZED',
      }),
    });
  });

  test('useSpot:false produces an On-Demand compute environment (reclaim-free runs)', () => {
    const oapp = new cdk.App();
    const obase = new SharedBaseStack(oapp, 'OBase', { env: ENV, namePrefix: 'pai' });
    const ostack = new RlPatternAStack(oapp, 'ORlPatternA', {
      env: ENV,
      namePrefix: 'pai',
      base: obase,
      useSpot: false,
    });
    const ot = Template.fromStack(ostack);
    ot.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      ComputeResources: Match.objectLike({
        Type: 'EC2', // On-Demand managed EC2 (not SPOT)
        InstanceTypes: ['g6e.4xlarge', 'g5.4xlarge'],
      }),
    });
  });

  test('creates a job queue and a GPU job definition (1 GPU)', () => {
    t.resourceCountIs('AWS::Batch::JobQueue', 1);
    t.resourceCountIs('AWS::Batch::JobDefinition', 2); // Isaac Lab PPO + VLA RLVR (RLinf)
    t.hasResourceProperties('AWS::Batch::JobDefinition', {
      ContainerProperties: Match.objectLike({
        ResourceRequirements: Match.arrayWith([
          Match.objectLike({ Type: 'GPU', Value: '1' }),
        ]),
      }),
    });
  });

  test('the RL job definition mounts the shared EFS at /mnt/efs with transit encryption', () => {
    t.hasResourceProperties('AWS::Batch::JobDefinition', {
      ContainerProperties: Match.objectLike({
        MountPoints: Match.arrayWith([
          Match.objectLike({ ContainerPath: '/mnt/efs' }),
        ]),
        Volumes: Match.arrayWith([
          Match.objectLike({
            EfsVolumeConfiguration: Match.objectLike({
              TransitEncryption: 'ENABLED',
            }),
          }),
        ]),
      }),
    });
  });

  test('the job command injects rl_train_bootstrap zlib-compressed (python3 -c <stub>), not a baked entrypoint', () => {
    const def = jobDef('IsaacRlJobDef');
    const cmd = def.Properties.ContainerProperties.Command as string[];
    expect(cmd[0]).toBe('python3');
    expect(cmd[1]).toBe('-c');
    // The bootstrap is zlib-compressed at synth (the raw ~10.6 KB source exceeds Batch's
    // 8192-byte ECS container-override ceiling), so the command is a decompress-and-run
    // stub, NOT the raw source. It runs main() by exec-ing under __name__='__main__'.
    expect(cmd[2]).toContain('import base64,zlib');
    expect(cmd[2]).toContain('zlib.decompress');
    expect(cmd[2]).toContain('"__name__":"__main__"');
    expect(cmd[2]).not.toContain('Isaac Lab RL — AWS Batch bootstrap'); // raw banner must be compressed away
  });

  test('the FULL merged ECS container override stays under Batch\'s 8192-byte limit', () => {
    // Batch counts the WHOLE containerOverride at SubmitJob — command + the merged
    // environment + resourceRequirements — against ECS's 8192-byte ceiling, NOT the command
    // in isolation. (The GR00T sibling stack hit this for real: a 6986-byte command passed a
    // `cmd.length < 7000` gate but failed the live submit because the merged blob was 7765
    // bytes.) Reconstruct the worst-case rl_launch.py override so CI catches a regression.
    const def = jobDef('IsaacRlJobDef');
    const cmd = def.Properties.ContainerProperties.Command as string[];
    const rr = def.Properties.ContainerProperties.ResourceRequirements as any[];
    // ECS sees the JobDefinition defaults AND every key rl_launch.py can set, so the
    // worst case is their union with realistic long values. The old list left out
    // RL_PLAY_ENVS / RL_NUM_ENVS / RL_SEED / RL_EXTRA_OVERRIDES and the JD defaults: it
    // passed at ~7.6k while a plain `--max-iterations 100 --num-envs 4096` submit was
    // refused live with "Container Overrides length must be at most 8192".
    const jdEnv = (def.Properties.ContainerProperties.Environment as Array<{ Name: string; Value: any }>)
      .map((e) => ({ name: e.Name, value: typeof e.Value === 'string' ? e.Value : 's3://pai-artifacts-123456789012-us-west-2/isaac-rl' }));
    const worstEnv = [
      ...jdEnv,
      { name: 'RL_PLAY_ENVS', value: '32' },
      { name: 'RL_NUM_ENVS', value: '4096' },
      { name: 'RL_SEED', value: '42' },
      { name: 'RL_SKIP_EXPORT', value: 'true' },
      { name: 'RL_EXTRA_OVERRIDES', value: 'env.rewards.feet_air_time.weight=0.5 agent.algorithm.learning_rate=0.0003' },
      { name: 'RL_TASK', value: 'Isaac-Velocity-Rough-H1-v0' },
      { name: 'RL_EXPERIMENT_NAME', value: 'h1_rough' },
      { name: 'RL_OUTPUT_S3', value: 's3://pai-artifacts-123456789012-us-west-2/isaac-rl/isaac-rl-20260617-181938' },
      { name: 'RL_CHECKPOINT_DIR', value: '/mnt/efs/rl-checkpoints' },
      { name: 'RL_NUM_GPUS', value: '1' },
      { name: 'RL_ISAACLAB_DIR', value: '/workspace/IsaacLab' },
      { name: 'RL_MAX_ITERATIONS', value: '3000' },
      { name: 'RL_LIVENESS_DEADLINE_S', value: '1200' },
      { name: 'RL_HYDRA_OVERRIDES', value: 'env.scene.num_envs=4096 agent.max_iterations=3000' },
    ];
    // ECS merges by name (override wins), so count each key once.
    const env = [...new Map(worstEnv.map((e) => [e.name, e] as [string, typeof e])).values()];
    const merged = JSON.stringify({ command: cmd, environment: env, resourceRequirements: rr });
    expect(Buffer.byteLength(merged, 'utf8')).toBeLessThan(8000);
  });

  test('the embedded payload decompresses back to the exact rl_train_bootstrap source', () => {
    const fs = require('fs');
    const path = require('path');
    const zlib = require('zlib');
    const def = jobDef('IsaacRlJobDef');
    const cmd = def.Properties.ContainerProperties.Command as string[];
    // Extract the base64 payload from the stub and round-trip it.
    const m = cmd[2].match(/b64decode\("([^"]+)"\)/);
    expect(m).not.toBeNull();
    const restored = zlib.inflateSync(Buffer.from(m![1], 'base64')).toString('utf8');
    const orig = fs.readFileSync(
      path.join(__dirname, '..', 'containers', 'isaac-lab-rl', 'rl_train_bootstrap.py'),
      'utf8',
    );
    // Full-line comments are dropped at synth (lib/shared/py-bootstrap.ts); everything
    // else is byte-identical. AST equivalence is proven in test/shared.test.ts.
    expect(restored).toBe(stripPyCommentLines(orig));
    // sanity: the restored source really is the RL bootstrap
    expect(restored).toContain('Isaac Lab RL — AWS Batch bootstrap (RL Pattern A)');
    expect(restored).toContain('RL_TASK');
  });

  test('the job definition sets the RL_* contract defaults for the reference task', () => {
    const def = jobDef('IsaacRlJobDef');
    const envPairs = def.Properties.ContainerProperties.Environment as Array<{ Name: string; Value: any }>;
    const byName = Object.fromEntries(envPairs.map((e) => [e.Name, e.Value]));
    expect(byName['RL_TASK']).toBe('Isaac-Velocity-Rough-H1-v0');
    expect(byName['RL_EXPERIMENT_NAME']).toBe('h1_rough');
    expect(byName['RL_NUM_GPUS']).toBe('1');
    // RL_OUTPUT_S3 references the artifact bucket (a CFN token, not a literal).
    expect(byName['RL_OUTPUT_S3']).toBeDefined();
  });

  test('uses the isaac-lab-rl ECR repo, NOT the vla-ft repo', () => {
    const def = jobDef('IsaacRlJobDef');
    const image = JSON.stringify(def.Properties.ContainerProperties.Image);
    expect(image).toContain('IsaacLabRlRepo');
    expect(image).not.toContain('VlaFtRepo');
  });

  describe('VLA RLVR job definition (RLinf vendor image)', () => {
    const vrl = () => jobDef('VlaRlJobDef');

    test('pins the RLinf image by digest (no mutable tag) from Docker Hub', () => {
      const image = vrl().Properties.ContainerProperties.Image as string;
      expect(image).toMatch(/^docker\.io\/rlinf\/rlinf@sha256:[0-9a-f]{64}$/);
    });

    test('1 GPU, 110 GiB (fits only g6e.4xlarge), 20 GiB /dev/shm, shared EFS', () => {
      const cp = vrl().Properties.ContainerProperties;
      const rr = Object.fromEntries((cp.ResourceRequirements as any[]).map((r) => [r.Type, r.Value]));
      expect(rr).toMatchObject({ GPU: '1', MEMORY: String(110 * 1024), VCPU: '16' });
      expect(cp.LinuxParameters.SharedMemorySize).toBe(20 * 1024);
      expect(JSON.stringify(cp.MountPoints)).toContain('/mnt/efs');
    });

    test('defaults to inspect mode with a pinned RLinf commit and the pi0.5 LIBERO SFT checkpoint', () => {
      const env = Object.fromEntries(
        (vrl().Properties.ContainerProperties.Environment as any[]).map((e) => [e.Name, e.Value]),
      );
      expect(env.VRL_MODE).toBe('inspect');
      expect(env.VRL_RLINF_SHA).toMatch(/^[0-9a-f]{40}$/);
      expect(env.VRL_MODEL_REPO).toBe('RLinf/RLinf-Pi05-LIBERO-SFT');
      expect(env.VRL_CONFIG).toBe('libero_spatial_ppo_openpi_pi05');
    });

    test('the packed command decompresses to vla_rl_bootstrap.py and the worst-case override fits 8192 bytes', () => {
      const cp = vrl().Properties.ContainerProperties;
      const cmd = cp.Command as string[];
      const b64 = /b64decode\("([^"]+)"\)/.exec(cmd[2])![1];
      const restored = zlib.inflateSync(Buffer.from(b64, 'base64')).toString('utf8');
      const orig = fs.readFileSync(path.join(__dirname, '..', 'containers', 'vla-rl', 'vla_rl_bootstrap.py'), 'utf8');
      expect(restored).toBe(stripPyCommentLines(orig));
      const env = (cp.Environment as any[]).map((e) => ({
        name: e.Name,
        value: typeof e.Value === 'string' ? e.Value : 's3://pai-artifacts-123456789012-us-west-2/vla-rl',
      }));
      env.push({ name: 'VRL_OVERRIDES', value: 'runner.max_epochs=200 env.train.total_num_envs=32 env.train.rollout_epoch=2 actor.micro_batch_size=64 actor.global_batch_size=256 actor.enable_offload=True rollout.enable_offload=True actor.optim.lr=1e-6' });
      env.push({ name: 'VRL_RUN_NAME', value: 'vla-rl-libero-spatial-pi05-20261017-120000' });
      const merged = JSON.stringify({ command: cmd, environment: env, resourceRequirements: cp.ResourceRequirements });
      expect(Buffer.byteLength(merged, 'utf8')).toBeLessThan(8000);
    });
  });

  test('three IAM roles: instance (ec2), execution (ecs-tasks), job (ecs-tasks + jobBasePolicy)', () => {
    const roles = t.findResources('AWS::IAM::Role');
    const principals = Object.values(roles).map(
      (r: any) => r.Properties.AssumeRolePolicyDocument.Statement[0].Principal.Service,
    );
    expect(principals).toContain('ec2.amazonaws.com');
    expect(principals.filter((p: any) => p === 'ecs-tasks.amazonaws.com').length).toBe(2);
  });

  test('job role attaches the base jobBasePolicy (cross-stack import)', () => {
    const roles = t.findResources('AWS::IAM::Role');
    const importsJobBase = Object.values(roles).some((r: any) => {
      const arns = r.Properties.ManagedPolicyArns;
      return (
        Array.isArray(arns) &&
        arns.some((a: any) => a['Fn::ImportValue'] && String(JSON.stringify(a)).includes('JobBasePolicy'))
      );
    });
    expect(importsJobBase).toBe(true);
  });

  test('no extra dataset-read statement (RL has no dataset; simulator makes its own data)', () => {
    const policies = t.findResources('AWS::IAM::Policy');
    const hasExtra = Object.values(policies).some((p: any) =>
      (p.Properties.PolicyDocument.Statement as any[]).some((s) => s.Sid === 'ExtraDatasetRead'),
    );
    expect(hasExtra).toBe(false);
  });

  test('a launch template enlarges the root volume to 250 GB gp3 (no ImageId → Batch picks GPU AMI)', () => {
    t.hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: Match.objectLike({
        BlockDeviceMappings: Match.arrayWith([
          Match.objectLike({
            Ebs: Match.objectLike({ VolumeSize: 250, VolumeType: 'gp3', Encrypted: true }),
          }),
        ]),
      }),
    });
    const lts = t.findResources('AWS::EC2::LaunchTemplate');
    const lt = Object.values(lts)[0] as any;
    expect(lt.Properties.LaunchTemplateData.ImageId).toBeUndefined();
  });

  test('retry strategy: liveness-guard kill (exit 42) does NOT retry; Spot reclaim does', () => {
    // The bootstrap's fail-fast liveness guard exits 42 when the trainer booted but never
    // started learning (the 5.5 h idle-run class) — a deterministic fault, so EXIT (no
    // retry). A genuine Spot reclaim still RETRYs to resume from EFS.
    t.hasResourceProperties('AWS::Batch::JobDefinition', {
      RetryStrategy: Match.objectLike({
        Attempts: 2,
        EvaluateOnExit: Match.arrayWith([
          Match.objectLike({ Action: 'EXIT', OnExitCode: '42' }),
          Match.objectLike({ Action: 'RETRY', OnStatusReason: 'Host EC2*' }),
        ]),
      }),
    });
  });

  test('exposes the job queue + job definition + output ARNs as outputs', () => {
    t.hasOutput('JobQueueArn', {});
    t.hasOutput('JobDefinitionArn', {});
    t.hasOutput('OutputS3Hint', {});
  });

  test('the pattern stack never owns an SNS topic (Base owns the shared topic)', () => {
    t.resourceCountIs('AWS::SNS::Topic', 0);
  });

  test('omitting notifyEmail (Base has no topic) creates no EventBridge rule', () => {
    t.resourceCountIs('AWS::Events::Rule', 0);
  });

  describe('with notifyEmail (topic on Base, rule on the pattern stack)', () => {
    const napp = new cdk.App();
    const nbase = new SharedBaseStack(napp, 'NBase', {
      env: ENV,
      namePrefix: 'pai',
      notifyEmail: 'you@example.com',
    });
    const nstack = new RlPatternAStack(napp, 'NRlPatternA', {
      env: ENV,
      namePrefix: 'pai',
      base: nbase,
    });
    const nbaseT = Template.fromStack(nbase);
    const nt = Template.fromStack(nstack);

    test('Base owns the one SNS topic + email subscription; the pattern stack none', () => {
      nbaseT.resourceCountIs('AWS::SNS::Topic', 1);
      nbaseT.hasResourceProperties('AWS::SNS::Subscription', {
        Protocol: 'email',
        Endpoint: 'you@example.com',
      });
      nt.resourceCountIs('AWS::SNS::Topic', 0);
    });

    test('the pattern stack creates an EventBridge rule filtering Batch terminal states with the isaac-rl- prefix', () => {
      nt.hasResourceProperties('AWS::Events::Rule', {
        EventPattern: Match.objectLike({
          source: ['aws.batch'],
          'detail-type': ['Batch Job State Change'],
          detail: Match.objectLike({
            jobName: [{ prefix: 'isaac-rl-' }],
            status: ['FAILED', 'SUCCEEDED'],
          }),
        }),
      });
    });
  });
});
