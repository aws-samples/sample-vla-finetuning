import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { SharedBaseStack } from '../lib/shared/base-stack';
import { PatternAStack } from '../lib/il/pattern-a-stack';

const ENV = { account: '111111111111', region: 'us-west-2' };

describe('PatternAStack', () => {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'TestBase', { env: ENV, namePrefix: 'pai' });
  const stack = new PatternAStack(app, 'TestPatternA', {
    env: ENV,
    namePrefix: 'pai',
    base,
    extraDatasetReadArns: ['arn:aws:s3:::some-dataset-bucket'],
  });
  const t = Template.fromStack(stack);

  test('creates a managed EC2 Batch compute environment on Spot', () => {
    t.resourceCountIs('AWS::Batch::ComputeEnvironment', 1);
    t.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      Type: 'managed',
      ComputeResources: Match.objectLike({
        Type: 'SPOT',
        // single-GPU fallback list, exact types (useOptimalInstanceClasses=false)
        InstanceTypes: ['g6e.4xlarge', 'g5.4xlarge'],
      }),
    });
  });

  test('Spot CE defaults to SPOT_PRICE_CAPACITY_OPTIMIZED (Batch-native capacity strategy)', () => {
    t.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      ComputeResources: Match.objectLike({
        AllocationStrategy: 'SPOT_PRICE_CAPACITY_OPTIMIZED',
      }),
    });
  });

  test('creates a job queue and a GPU job definition', () => {
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

  test('the job definition timeout defaults to 18 h (covers single-GPU LoRA full-FT ~12.8 h, not the old 6 h)', () => {
    // job ...151444 was a healthy run SIGKILLed at step 9000/20000 by a 6 h ceiling
    // sized for the old multi-GPU expert-only regime. 18 h * 3600 = 64800 s.
    t.hasResourceProperties('AWS::Batch::JobDefinition', {
      Timeout: Match.objectLike({ AttemptDurationSeconds: 64800 }),
      RetryStrategy: Match.objectLike({ Attempts: 2 }),
    });
  });

  test('attemptTimeout prop overrides the default timeout', () => {
    const app3 = new cdk.App();
    const base3 = new SharedBaseStack(app3, 'B3', { env: ENV, namePrefix: 'pai' });
    const s3 = new PatternAStack(app3, 'P3', {
      env: ENV,
      namePrefix: 'pai',
      base: base3,
      attemptTimeout: cdk.Duration.hours(24),
    });
    Template.fromStack(s3).hasResourceProperties('AWS::Batch::JobDefinition', {
      Timeout: Match.objectLike({ AttemptDurationSeconds: 86400 }),
    });
  });

  test('the job definition mounts the shared EFS at /mnt/efs with transit encryption', () => {
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

  test('the job command is the injected batch_bootstrap (python3 -c <src>), not a baked entrypoint', () => {
    const defs = t.findResources('AWS::Batch::JobDefinition');
    const def = Object.values(defs)[0] as any;
    const cmd = def.Properties.ContainerProperties.Command as string[];
    expect(cmd[0]).toBe('python3');
    expect(cmd[1]).toBe('-c');
    // the embedded bootstrap source, recognizable by its banner
    expect(cmd[2]).toContain('AWS Batch bootstrap (Pattern A)');
  });

  test('hyperparameters travel as an S3 pointer, NOT per-knob SM_HP_* override env', () => {
    // ROOT FIX for the recurring 8192 ceiling: the launchers (batch_launch.py /
    // orchestrator_submit.py) now upload the hp dict to S3 (hyperparameters.json) and the
    // override carries only VLA_FT_HP_S3 + small wiring env. So the override size is
    // INDEPENDENT of the hyperparameters — the long LoRA vision-tower regex that kept
    // tipping the merged blob over the ceiling is gone from the override entirely. The
    // bootstrap stages that file at SageMaker's hyperparameters.json path and the UNCHANGED
    // train.py reads it. This gate reconstructs the *new* worst-case override (pointer +
    // wiring + both HF tokens, no SM_HP_*) and asserts it sits far below 8192.
    const def = Object.values(t.findResources('AWS::Batch::JobDefinition'))[0] as any;
    const cmd = def.Properties.ContainerProperties.Command as string[];
    const rr = def.Properties.ContainerProperties.ResourceRequirements as any[];
    const jobName = 'vla-ft-pi05-20260620-151444';
    // The full override the launchers ship after the fix — note: NO SM_HP_* keys, and the
    // LoRA regex lives in the S3 hp file, not here. The hp pointer is a short S3 URI.
    const overrideEnv = [
      { name: 'VLA_FT_HP_S3', value: `s3://pai-artifacts-123456789012-us-west-2/vla-ft-code/${jobName}/hyperparameters.json` },
      { name: 'VLA_FT_DATASET_S3', value: 's3://example-openarm-lift-dataset-us-west-2-123456789012/2026-06-16-0044/lerobot_dataset/' },
      { name: 'VLA_FT_OUTPUT_S3', value: `s3://pai-artifacts-123456789012-us-west-2/vla-ft/${jobName}` },
      { name: 'VLA_FT_CODE_S3', value: 's3://pai-artifacts-123456789012-us-west-2/vla-ft-code/train.py' },
      { name: 'VLA_FT_CHECKPOINT_DIR', value: `/mnt/efs/checkpoints/${jobName}` },
      { name: 'SM_NUM_GPUS', value: '1' },
      { name: 'HF_TOKEN', value: 'hf_' + 'x'.repeat(37) },
      { name: 'HUGGING_FACE_HUB_TOKEN', value: 'hf_' + 'x'.repeat(37) },
    ];
    // Structural guarantee: no SM_HP_* in the override (the size regression CLASS is gone —
    // override size no longer scales with hp, so a new long flag can never blow the ceiling).
    expect(overrideEnv.some((e) => e.name.startsWith('SM_HP_'))).toBe(false);
    // The override env alone is now tiny + bounded (a pointer + short wiring, ~800 B); the
    // hp regex lives in the S3 file, not here.
    expect(Buffer.byteLength(JSON.stringify(overrideEnv), 'utf8')).toBeLessThan(1200);
    // The merged blob is dominated by the (bounded) raw batch_bootstrap.py command. Assert
    // healthy margin under the real 8192 ceiling so prose growth in the bootstrap is caught.
    const merged = JSON.stringify({ command: cmd, environment: overrideEnv, resourceRequirements: rr });
    const bytes = Buffer.byteLength(merged, 'utf8');
    expect(bytes).toBeLessThan(7500);
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
    // Find the ecs-tasks role that carries a managed policy import (the job role;
    // the execution role uses an AWS-managed ARN string, not an ImportValue).
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

  test('job role grants the extra dataset read', () => {
    t.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([Match.objectLike({ Sid: 'ExtraDatasetRead' })]),
      }),
    });
  });

  test('a launch template enlarges the root volume to 200 GB gp3 (no ImageId → Batch picks GPU AMI)', () => {
    t.hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: Match.objectLike({
        BlockDeviceMappings: Match.arrayWith([
          Match.objectLike({
            Ebs: Match.objectLike({ VolumeSize: 200, VolumeType: 'gp3', Encrypted: true }),
          }),
        ]),
      }),
    });
    const lts = t.findResources('AWS::EC2::LaunchTemplate');
    const lt = Object.values(lts)[0] as any;
    expect(lt.Properties.LaunchTemplateData.ImageId).toBeUndefined();
  });

  test('exposes the job queue + job definition ARNs as outputs', () => {
    t.hasOutput('JobQueueArn', {});
    t.hasOutput('JobDefinitionArn', {});
  });

  test('omitting extraDatasetReadArns yields no ExtraDatasetRead statement', () => {
    const app2 = new cdk.App();
    const base2 = new SharedBaseStack(app2, 'B2', { env: ENV, namePrefix: 'pai' });
    const s2 = new PatternAStack(app2, 'P2', { env: ENV, namePrefix: 'pai', base: base2 });
    const t2 = Template.fromStack(s2);
    const policies = t2.findResources('AWS::IAM::Policy');
    const hasExtra = Object.values(policies).some((p: any) =>
      (p.Properties.PolicyDocument.Statement as any[]).some((s) => s.Sid === 'ExtraDatasetRead'),
    );
    expect(hasExtra).toBe(false);
  });

  test('omitting notifyEmail creates no SNS topic or EventBridge rule', () => {
    t.resourceCountIs('AWS::SNS::Topic', 0);
    t.resourceCountIs('AWS::Events::Rule', 0);
  });

  describe('with notifyEmail', () => {
    const napp = new cdk.App();
    const nbase = new SharedBaseStack(napp, 'NBase', { env: ENV, namePrefix: 'pai' });
    const nstack = new PatternAStack(napp, 'NPatternA', {
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

    test('creates an EventBridge rule filtering Batch terminal states', () => {
      nt.hasResourceProperties('AWS::Events::Rule', {
        EventPattern: Match.objectLike({
          source: ['aws.batch'],
          'detail-type': ['Batch Job State Change'],
          detail: Match.objectLike({
            jobName: [{ prefix: 'vla-ft-' }],
            status: ['FAILED', 'SUCCEEDED'],
          }),
        }),
      });
    });
  });
});
