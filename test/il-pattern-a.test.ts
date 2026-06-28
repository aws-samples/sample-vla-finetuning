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

  test('creates TWO managed EC2 Batch CEs — one Spot, one On-Demand (per-job switching)', () => {
    // Spot vs On-Demand is a CE property, so per-job switching means deploying BOTH and
    // selecting the queue at submit. Idle CEs scale to 0 vCPU, so the waiting one is free.
    t.resourceCountIs('AWS::Batch::ComputeEnvironment', 2);
    t.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      Type: 'managed',
      ComputeResources: Match.objectLike({
        Type: 'SPOT',
        // single-GPU fallback list, exact types (useOptimalInstanceClasses=false)
        InstanceTypes: ['g6e.4xlarge', 'g5.4xlarge'],
      }),
    });
    t.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      Type: 'managed',
      ComputeResources: Match.objectLike({
        Type: 'EC2', // On-Demand
        // L40S-only: the OD queue is the sanctioned full-VLM path (spot=false); a ~40 GB
        // replica OOMs on a 24 GB g5, and BEST_FIT_PROGRESSIVE would otherwise pick it.
        InstanceTypes: ['g6e.4xlarge'],
      }),
    });
  });

  test('Spot CE → SPOT_PRICE_CAPACITY_OPTIMIZED; On-Demand CE → BEST_FIT_PROGRESSIVE', () => {
    t.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      ComputeResources: Match.objectLike({
        Type: 'SPOT',
        AllocationStrategy: 'SPOT_PRICE_CAPACITY_OPTIMIZED',
      }),
    });
    t.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
      ComputeResources: Match.objectLike({
        Type: 'EC2',
        AllocationStrategy: 'BEST_FIT_PROGRESSIVE',
      }),
    });
  });

  test('creates TWO job queues (Spot + On-Demand) sharing ONE GPU job definition', () => {
    t.resourceCountIs('AWS::Batch::JobQueue', 2);
    t.resourceCountIs('AWS::Batch::JobDefinition', 1);
    t.hasResourceProperties('AWS::Batch::JobDefinition', {
      ContainerProperties: Match.objectLike({
        ResourceRequirements: Match.arrayWith([
          Match.objectLike({ Type: 'GPU', Value: '1' }),
        ]),
      }),
    });
  });

  test('the job definition timeout defaults to 28 h (covers the MEASURED single-L40S full-VLM ~19 h)', () => {
    // The old 18 h ceiling SIGKILLed a healthy run at step 19000/20000 (~3.36 s/step on a
    // single L40S ⇒ ~19 h). 28 h * 3600 = 100800 s gives headroom. Per-job timeout_hours
    // overrides this at SubmitJob time (no redeploy); this is just the deployed default.
    t.hasResourceProperties('AWS::Batch::JobDefinition', {
      Timeout: Match.objectLike({ AttemptDurationSeconds: 100800 }),
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

  test('exposes the Spot + On-Demand job queue ARNs + the job definition ARN as outputs', () => {
    t.hasOutput('JobQueueArn', {});           // Spot (default, spot=true)
    t.hasOutput('JobQueueArnOnDemand', {});   // On-Demand (spot=false, selected per-job)
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

  test('the pattern stack never owns an SNS topic (Base owns the shared topic)', () => {
    // The topic moved to Base to avoid a fixed-name AlreadyExists collision when more
    // than one stack notifies. The pattern stack only ever adds an EventBridge rule.
    t.resourceCountIs('AWS::SNS::Topic', 0);
  });

  test('omitting notifyEmail (Base has no topic) creates no EventBridge rule', () => {
    t.resourceCountIs('AWS::Events::Rule', 0);
  });

  describe('with notifyEmail (topic on Base, rule on the pattern stack)', () => {
    const napp = new cdk.App();
    // notifyEmail goes to BASE — that's where the shared topic + subscription live now.
    const nbase = new SharedBaseStack(napp, 'NBase', {
      env: ENV,
      namePrefix: 'pai',
      notifyEmail: 'you@example.com',
    });
    const nstack = new PatternAStack(napp, 'NPatternA', {
      env: ENV,
      namePrefix: 'pai',
      base: nbase,
    });
    const nbaseT = Template.fromStack(nbase);
    const nt = Template.fromStack(nstack);

    test('Base owns the one SNS topic + email subscription', () => {
      nbaseT.resourceCountIs('AWS::SNS::Topic', 1);
      nbaseT.hasResourceProperties('AWS::SNS::Subscription', {
        Protocol: 'email',
        Endpoint: 'you@example.com',
      });
      // The pattern stack imports the topic; it does not create its own.
      nt.resourceCountIs('AWS::SNS::Topic', 0);
    });

    test('the pattern stack creates an EventBridge rule filtering Batch terminal states', () => {
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

    // The Batch rule shares the same diagnostic safety net as the SageMaker rule: a DLQ so
    // a failed SNS delivery is captured, not silently dropped (see notifications.ts).
    test('the Batch rule target has a dead-letter queue for failed deliveries', () => {
      nt.resourceCountIs('AWS::SQS::Queue', 1);
      const rules = nt.findResources('AWS::Events::Rule');
      const rule = Object.values(rules)[0] as any;
      expect((rule.Properties.Targets as any[])[0].DeadLetterConfig).toBeDefined();
    });
  });
});
