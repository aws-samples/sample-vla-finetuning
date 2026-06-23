import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { SharedBaseStack } from '../lib/shared/base-stack';
import { GrootPatternAStack } from '../lib/il/gr00t-pattern-a-stack';

const ENV = { account: '111111111111', region: 'us-west-2' };
const GROOT_G1_ARN = 'arn:aws:s3:::example-gr00t-g1-dataset';

describe('GrootPatternAStack', () => {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'TestBase', { env: ENV, namePrefix: 'pai' });
  const stack = new GrootPatternAStack(app, 'TestGrootPatternA', {
    env: ENV,
    namePrefix: 'pai',
    base,
    extraDatasetReadArns: [GROOT_G1_ARN],
  });
  const t = Template.fromStack(stack);

  test('default compute environment is On-Demand (EC2), g6e L40S only — NO g5 fallback', () => {
    t.resourceCountIs('AWS::Batch::ComputeEnvironment', 1);
    t.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      Type: 'managed',
      ComputeResources: Match.objectLike({
        Type: 'EC2', // On-Demand by default (no EFS resume → reclaim would restart)
        InstanceTypes: ['g6e.4xlarge'], // L40S only; A10G (g5) OOMs a 3B Cosmos model
      }),
    });
    // Defensive: the g5 fallback that the lerobot Pattern A uses must NOT be here.
    const ces = t.findResources('AWS::Batch::ComputeEnvironment');
    const ce = Object.values(ces)[0] as any;
    expect(ce.Properties.ComputeResources.InstanceTypes).not.toContain('g5.4xlarge');
  });

  test('useSpot:true produces a Spot compute environment (documented opt-in)', () => {
    const sapp = new cdk.App();
    const sbase = new SharedBaseStack(sapp, 'SBase', { env: ENV, namePrefix: 'pai' });
    const sstack = new GrootPatternAStack(sapp, 'SGrootPatternA', {
      env: ENV,
      namePrefix: 'pai',
      base: sbase,
      useSpot: true,
    });
    const st = Template.fromStack(sstack);
    st.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      ComputeResources: Match.objectLike({
        Type: 'SPOT',
        AllocationStrategy: 'SPOT_PRICE_CAPACITY_OPTIMIZED',
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

  test('the job command injects gr00t_train_bootstrap zlib-compressed (python3 -c <stub>)', () => {
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    const cmd = def.Properties.ContainerProperties.Command as string[];
    expect(cmd[0]).toBe('python3');
    expect(cmd[1]).toBe('-c');
    expect(cmd[2]).toContain('import base64,zlib');
    expect(cmd[2]).toContain('zlib.decompress');
    expect(cmd[2]).toContain('"__name__":"__main__"');
    // raw banner must be compressed away (not present as plaintext in the stub)
    expect(cmd[2]).not.toContain('GR00T N1.7 fine-tune - AWS Batch bootstrap');
  });

  test('the FULL merged ECS container override stays under Batch\'s 8192-byte limit', () => {
    // Batch counts the WHOLE containerOverride at SubmitJob — command + the merged
    // environment + resourceRequirements — against ECS's 8192-byte ceiling, NOT the command
    // in isolation. A 6986-byte command passed the old `cmd[2].length < 7000` gate yet failed
    // the real submit ("Container Overrides length must be at most 8192") because the merged
    // blob was 7765 bytes. This gate reconstructs that worst-case blob so CI catches it.
    const def = Object.values(t.findResources('AWS::Batch::JobDefinition'))[0] as any;
    const cmd = def.Properties.ContainerProperties.Command as string[];
    const rr = def.Properties.ContainerProperties.ResourceRequirements as any[];
    // Worst-case launcher env (gr00t_launch.py): every GROOT_* key the wrapper can set,
    // with the longest realistic values (the G1 dataset URI is ~95 chars) + the optional
    // HF token. The launcher REPLACES the baked env, so this is the override that ships.
    const worstEnv = [
      { name: 'GROOT_DATASET_S3', value: 's3://example-gr00t-g1-dataset-us-west-2-123456789012/2026-06-17-0410/task-0/lerobot/' },
      { name: 'GROOT_OUTPUT_S3', value: 's3://pai-artifacts-123456789012-us-west-2/gr00t-n17/gr00t-n17-20260618-162028' },
      { name: 'GROOT_BASE_MODEL', value: 'nvidia/GR00T-N1.7-3B' },
      { name: 'GROOT_EMBODIMENT_TAG', value: 'UNITREE_G1' },
      { name: 'GROOT_MAX_STEPS', value: '20000' },
      { name: 'GROOT_SAVE_STEPS', value: '1000' },
      { name: 'GROOT_GLOBAL_BATCH', value: '64' },
      { name: 'GROOT_NUM_GPUS', value: '1' },
      { name: 'GROOT_LEARNING_RATE', value: '1e-4' },
      { name: 'GROOT_LIVENESS_DEADLINE_S', value: '2400' },
      { name: 'GROOT_ACTION_HORIZON', value: '50' },
      { name: 'GROOT_MODALITY_CONFIG', value: '/workspace/gr00t/experiment/data_config.py' },
      { name: 'GROOT_EXTRA_ARGS', value: '--weight-decay 1e-4 --tune-llm' },
      { name: 'HF_TOKEN', value: 'hf_' + 'x'.repeat(37) },
      { name: 'HUGGING_FACE_HUB_TOKEN', value: 'hf_' + 'x'.repeat(37) },
    ];
    const merged = JSON.stringify({ command: cmd, environment: worstEnv, resourceRequirements: rr });
    const bytes = Buffer.byteLength(merged, 'utf8');
    // Assert against the real 8192 ceiling with margin (the empirically-safe stacks sit ~6.6k).
    expect(bytes).toBeLessThan(8000);
  });

  test('the embedded payload decompresses back to the exact gr00t_train_bootstrap source', () => {
    const fs = require('fs');
    const path = require('path');
    const zlib = require('zlib');
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    const cmd = def.Properties.ContainerProperties.Command as string[];
    const m = cmd[2].match(/b64decode\("([^"]+)"\)/);
    expect(m).not.toBeNull();
    const restored = zlib.inflateSync(Buffer.from(m![1], 'base64')).toString('utf8');
    const orig = fs.readFileSync(
      path.join(__dirname, '..', 'containers', 'gr00t-n17', 'gr00t_train_bootstrap.py'),
      'utf8',
    );
    expect(restored).toBe(orig);
    expect(restored).toContain('GR00T N1.7 fine-tune - AWS Batch bootstrap (IL GR00T Pattern A)');
    expect(restored).toContain('GROOT_DATASET_S3');
  });

  test('the job definition sets the GROOT_* contract defaults for the G1 reference run', () => {
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    const envPairs = def.Properties.ContainerProperties.Environment as Array<{ Name: string; Value: any }>;
    const byName = Object.fromEntries(envPairs.map((e) => [e.Name, e.Value]));
    expect(byName['GROOT_BASE_MODEL']).toBe('nvidia/GR00T-N1.7-3B');
    expect(byName['GROOT_EMBODIMENT_TAG']).toBe('UNITREE_G1');
    expect(byName['GROOT_NUM_GPUS']).toBe('1');
    expect(byName['GROOT_OUTPUT_S3']).toBeDefined();
  });

  test('uses the gr00t-n17 ECR repo, NOT the vla-ft or isaac-lab-rl repos', () => {
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    const image = JSON.stringify(def.Properties.ContainerProperties.Image);
    expect(image).toContain('GrootRepo');
    expect(image).not.toContain('VlaFtRepo');
    expect(image).not.toContain('IsaacLabRlRepo');
  });

  test('three IAM roles: instance (ec2), execution (ecs-tasks), job (ecs-tasks + jobBasePolicy)', () => {
    const roles = t.findResources('AWS::IAM::Role');
    const principals = Object.values(roles).map(
      (r: any) => r.Properties.AssumeRolePolicyDocument.Statement[0].Principal.Service,
    );
    expect(principals).toContain('ec2.amazonaws.com');
    expect(principals.filter((p: any) => p === 'ecs-tasks.amazonaws.com').length).toBe(2);
  });

  test('job role gets an ExtraDatasetRead statement for the GR00T-G1 bucket', () => {
    const policies = t.findResources('AWS::IAM::Policy');
    const hasExtra = Object.values(policies).some((p: any) =>
      (p.Properties.PolicyDocument.Statement as any[]).some((s) => s.Sid === 'ExtraDatasetRead'),
    );
    expect(hasExtra).toBe(true);
  });

  test('sets a 16 GiB /dev/shm (LinuxParameters.SharedMemorySize) for the DataLoader workers', () => {
    // The GR00T DataLoader passes large multi-camera image tensors between workers via
    // /dev/shm; the 64 MiB container default causes a "Bus error / out of shared memory"
    // crash at step 0. SharedMemorySize is in MiB (16 GiB = 16384).
    t.hasResourceProperties('AWS::Batch::JobDefinition', {
      ContainerProperties: Match.objectLike({
        LinuxParameters: Match.objectLike({ SharedMemorySize: 16384 }),
      }),
    });
  });

  test('no EFS mount (checkpoints go to the local root volume, not EFS)', () => {
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    expect(def.Properties.ContainerProperties.Volumes).toBeUndefined();
    expect(def.Properties.ContainerProperties.MountPoints).toBeUndefined();
    // and the stack creates no standalone EFS-ingress rule
    t.resourceCountIs('AWS::EC2::SecurityGroupIngress', 0);
  });

  test('a launch template enlarges the root volume to 300 GB gp3 (no ImageId → Batch picks GPU AMI)', () => {
    t.hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: Match.objectLike({
        BlockDeviceMappings: Match.arrayWith([
          Match.objectLike({
            Ebs: Match.objectLike({ VolumeSize: 300, VolumeType: 'gp3', Encrypted: true }),
          }),
        ]),
      }),
    });
    const lts = t.findResources('AWS::EC2::LaunchTemplate');
    const lt = Object.values(lts)[0] as any;
    expect(lt.Properties.LaunchTemplateData.ImageId).toBeUndefined();
  });

  test('retry strategy: liveness-guard kill (exit 42) does NOT retry; Spot reclaim does', () => {
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
    t.hasOutput('ImageUriHint', {});
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
    const nstack = new GrootPatternAStack(napp, 'NGrootPatternA', {
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

    test('the pattern stack creates an EventBridge rule filtering Batch terminal states with the gr00t-n17- prefix', () => {
      nt.hasResourceProperties('AWS::Events::Rule', {
        EventPattern: Match.objectLike({
          source: ['aws.batch'],
          'detail-type': ['Batch Job State Change'],
          detail: Match.objectLike({
            jobName: [{ prefix: 'gr00t-n17-' }],
            status: ['FAILED', 'SUCCEEDED'],
          }),
        }),
      });
    });
  });
});
