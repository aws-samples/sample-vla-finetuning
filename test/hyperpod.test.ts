import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { SharedBaseStack } from '../lib/shared/base-stack';
import { IlHyperPodStack } from '../lib/il/hyperpod-stack';
import { RlHyperPodStack } from '../lib/rl/hyperpod-stack';

const ENV = { account: '111111111111', region: 'us-west-2' };

// Pattern C verification = code + `cdk synth` only (real multi-node deploy deferred).
// These tests ARE that gate: they prove both HyperPod stacks synthesize to a complete,
// valid CfnCluster template. The stacks are intentionally NOT in bin/app.ts (deploy
// gated), so instantiating them here is also the synth check.

describe('IlHyperPodStack', () => {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'TestBase', { env: ENV, namePrefix: 'pai' });
  const stack = new IlHyperPodStack(app, 'TestIlHyperPod', { env: ENV, namePrefix: 'pai', base });
  const t = Template.fromStack(stack);

  test('synthesizes exactly one HyperPod CfnCluster', () => {
    t.resourceCountIs('AWS::SageMaker::Cluster', 1);
  });

  test('cluster uses the Slurm orchestrator (self-contained at synth)', () => {
    t.hasResourceProperties('AWS::SageMaker::Cluster', {
      Orchestrator: { Slurm: Match.anyValue() },
    });
  });

  test('default IL instance group is multi-node L40S with a lifecycle config', () => {
    t.hasResourceProperties('AWS::SageMaker::Cluster', {
      InstanceGroups: Match.arrayWith([
        Match.objectLike({
          InstanceType: 'ml.g6e.48xlarge',
          InstanceCount: 2,
          LifeCycleConfig: Match.objectLike({ OnCreate: 'on_create.sh' }),
        }),
      ]),
    });
  });

  test('lifecycle S3 URI uses the HyperPod-required sagemaker- prefix', () => {
    const clusters = t.findResources('AWS::SageMaker::Cluster');
    const cluster = Object.values(clusters)[0] as any;
    const uri = cluster.Properties.InstanceGroups[0].LifeCycleConfig.SourceS3Uri as string;
    expect(uri.startsWith('s3://sagemaker-')).toBe(true);
  });

  test('cluster role attaches the HyperPod managed policy + base jobBasePolicy', () => {
    const roles = t.findResources('AWS::IAM::Role');
    const hyperpodRole = Object.values(roles).find(
      (r: any) => r.Properties.AssumeRolePolicyDocument.Statement[0].Principal.Service === 'sagemaker.amazonaws.com',
    ) as any;
    expect(hyperpodRole).toBeDefined();
    const arns = JSON.stringify(hyperpodRole.Properties.ManagedPolicyArns);
    expect(arns).toContain('AmazonSageMakerClusterInstanceRolePolicy');
    expect(arns).toContain('JobBasePolicy');
  });

  test('node recovery is Automatic (HyperPod resilience)', () => {
    t.hasResourceProperties('AWS::SageMaker::Cluster', { NodeRecovery: 'Automatic' });
  });
});

describe('RlHyperPodStack', () => {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'RBase', { env: ENV, namePrefix: 'pai' });
  const stack = new RlHyperPodStack(app, 'TestRlHyperPod', { env: ENV, namePrefix: 'pai', base });
  const t = Template.fromStack(stack);

  test('synthesizes one HyperPod CfnCluster with the RL default instance group', () => {
    t.resourceCountIs('AWS::SageMaker::Cluster', 1);
    t.hasResourceProperties('AWS::SageMaker::Cluster', {
      InstanceGroups: Match.arrayWith([
        Match.objectLike({ InstanceType: 'ml.g6.12xlarge', InstanceCount: 2 }),
      ]),
    });
  });

  test('custom instance groups override the default', () => {
    const app2 = new cdk.App();
    const base2 = new SharedBaseStack(app2, 'RB2', { env: ENV, namePrefix: 'pai' });
    const s2 = new RlHyperPodStack(app2, 'RlCustom', {
      env: ENV,
      namePrefix: 'pai',
      base: base2,
      instanceGroups: [{ name: 'worker', instanceType: 'ml.p5.48xlarge', instanceCount: 4 }],
    });
    const t2 = Template.fromStack(s2);
    t2.hasResourceProperties('AWS::SageMaker::Cluster', {
      InstanceGroups: Match.arrayWith([
        Match.objectLike({ InstanceType: 'ml.p5.48xlarge', InstanceCount: 4 }),
      ]),
    });
  });
});
