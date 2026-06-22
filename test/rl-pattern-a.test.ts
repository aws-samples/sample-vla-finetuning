import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { SharedBaseStack } from '../lib/shared/base-stack';
import { RlPatternAStack } from '../lib/rl/pattern-a-stack';

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
    t.resourceCountIs('AWS::Batch::JobDefinition', 1);
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
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
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
    const def = Object.values(t.findResources('AWS::Batch::JobDefinition'))[0] as any;
    const cmd = def.Properties.ContainerProperties.Command as string[];
    const rr = def.Properties.ContainerProperties.ResourceRequirements as any[];
    const worstEnv = [
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
    const merged = JSON.stringify({ command: cmd, environment: worstEnv, resourceRequirements: rr });
    expect(Buffer.byteLength(merged, 'utf8')).toBeLessThan(8000);
  });

  test('the embedded payload decompresses back to the exact rl_train_bootstrap source', () => {
    const fs = require('fs');
    const path = require('path');
    const zlib = require('zlib');
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    const cmd = def.Properties.ContainerProperties.Command as string[];
    // Extract the base64 payload from the stub and round-trip it.
    const m = cmd[2].match(/b64decode\("([^"]+)"\)/);
    expect(m).not.toBeNull();
    const restored = zlib.inflateSync(Buffer.from(m![1], 'base64')).toString('utf8');
    const orig = fs.readFileSync(
      path.join(__dirname, '..', 'containers', 'isaac-lab-rl', 'rl_train_bootstrap.py'),
      'utf8',
    );
    expect(restored).toBe(orig);
    // sanity: the restored source really is the RL bootstrap
    expect(restored).toContain('Isaac Lab RL — AWS Batch bootstrap (RL Pattern A)');
    expect(restored).toContain('RL_TASK');
  });

  test('the job definition sets the RL_* contract defaults for the reference task', () => {
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    const envPairs = def.Properties.ContainerProperties.Environment as Array<{ Name: string; Value: any }>;
    const byName = Object.fromEntries(envPairs.map((e) => [e.Name, e.Value]));
    expect(byName['RL_TASK']).toBe('Isaac-Velocity-Rough-H1-v0');
    expect(byName['RL_EXPERIMENT_NAME']).toBe('h1_rough');
    expect(byName['RL_NUM_GPUS']).toBe('1');
    // RL_OUTPUT_S3 references the artifact bucket (a CFN token, not a literal).
    expect(byName['RL_OUTPUT_S3']).toBeDefined();
  });

  test('uses the isaac-lab-rl ECR repo, NOT the vla-ft repo', () => {
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    const image = JSON.stringify(def.Properties.ContainerProperties.Image);
    expect(image).toContain('IsaacLabRlRepo');
    expect(image).not.toContain('VlaFtRepo');
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

  test('omitting notifyEmail creates no SNS topic or EventBridge rule', () => {
    t.resourceCountIs('AWS::SNS::Topic', 0);
    t.resourceCountIs('AWS::Events::Rule', 0);
  });

  describe('with notifyEmail', () => {
    const napp = new cdk.App();
    const nbase = new SharedBaseStack(napp, 'NBase', { env: ENV, namePrefix: 'pai' });
    const nstack = new RlPatternAStack(napp, 'NRlPatternA', {
      env: ENV,
      namePrefix: 'pai',
      base: nbase,
      notifyEmail: 'you@example.com',
    });
    const nt = Template.fromStack(nstack);

    test('creates an SNS topic with an email subscription', () => {
      nt.resourceCountIs('AWS::SNS::Topic', 1);
      nt.hasResourceProperties('AWS::SNS::Subscription', {
        Protocol: 'email',
        Endpoint: 'you@example.com',
      });
    });

    test('creates an EventBridge rule filtering Batch terminal states with the isaac-rl- prefix', () => {
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
