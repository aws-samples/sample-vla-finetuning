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

  test('cluster has a dedicated Slurm controller group + the GPU worker group (Slurm needs both)', () => {
    const clusters = t.findResources('AWS::SageMaker::Cluster');
    const groups = (Object.values(clusters)[0] as any).Properties.InstanceGroups as any[];
    const controller = groups.find((g) => g.InstanceGroupName === 'controller-machine');
    const worker = groups.find((g) => g.InstanceGroupName === 'worker-group-1');
    expect(controller).toBeDefined();
    expect(controller.InstanceType).toBe('ml.c5.2xlarge');
    expect(controller.InstanceCount).toBe(1);
    expect(worker).toBeDefined();
    expect(worker.InstanceType).toBe('ml.g6e.48xlarge');
    expect(worker.InstanceCount).toBe(2);
    // SlurmConfig node types: CREATE fails ("no InstanceGroup with Controller node type")
    // unless exactly one group is Controller and the GPU group is Compute. Regression lock
    // for the 1-shot deploy gap.
    expect(controller.SlurmConfig.NodeType).toBe('Controller');
    expect(worker.SlurmConfig.NodeType).toBe('Compute');
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

  test('cluster role grants VpcConfig ec2 permissions (DescribeSubnets etc.)', () => {
    // Regression lock: the managed instance-role policy has NO ec2 actions, so a
    // VPC-configured cluster fails CREATE with "Unable to retrieve subnets" unless the
    // execution role also carries the awslabs AdditionToEnableVpcConfig statement. It
    // renders as an inline policy on the Role (Properties.Policies), not a standalone
    // AWS::IAM::Policy — assert it there so this 1-shot deploy gap can't silently reappear.
    const roles = t.findResources('AWS::IAM::Role');
    const hyperpodRole = Object.values(roles).find(
      (r: any) => r.Properties.AssumeRolePolicyDocument.Statement[0].Principal.Service === 'sagemaker.amazonaws.com',
    ) as any;
    const inline = JSON.stringify(hyperpodRole.Properties.Policies);
    expect(inline).toContain('ec2:DescribeSubnets');
    expect(inline).toContain('ec2:DescribeVpcs');
    expect(inline).toContain('ec2:CreateNetworkInterface');
  });

  test('cluster SG has the EFA-required self-referencing egress rule', () => {
    // Regression lock: EFA RDMA needs an all-traffic egress rule targeting the SG by
    // reference (not the 0.0.0.0/0 CIDR). Without it cross-node NCCL all-reduce fails with
    // "Unreachable remote". Emitted as a standalone CfnSecurityGroupEgress (the L2 form is
    // silently dropped under allowAllOutbound:true).
    const egress = t.findResources('AWS::EC2::SecurityGroupEgress');
    const selfRef = Object.values(egress).find(
      (e: any) => e.Properties.IpProtocol === '-1' && e.Properties.DestinationSecurityGroupId,
    );
    expect(selfRef).toBeDefined();
  });

  test('node recovery is Automatic (HyperPod resilience)', () => {
    t.hasResourceProperties('AWS::SageMaker::Cluster', { NodeRecovery: 'Automatic' });
  });

  test('no FSx by default (attachFsx omitted → no continuously-billing filesystem)', () => {
    t.resourceCountIs('AWS::FSx::FileSystem', 0);
    t.resourceCountIs('AWS::FSx::DataRepositoryAssociation', 0);
  });
});

describe('IlHyperPodStack with attachFsx (the multi-node data plane)', () => {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'FBase', { env: ENV, namePrefix: 'pai' });
  const stack = new IlHyperPodStack(app, 'FsxHyperPod', {
    env: ENV,
    namePrefix: 'pai',
    base,
    attachFsx: true,
  });
  const t = Template.fromStack(stack);

  test('attaches ONE PERSISTENT_2 FSx Lustre + a bidirectional DRA linked to the dataBucket', () => {
    t.resourceCountIs('AWS::FSx::FileSystem', 1);
    t.hasResourceProperties('AWS::FSx::FileSystem', {
      FileSystemType: 'LUSTRE',
      LustreConfiguration: Match.objectLike({ DeploymentType: 'PERSISTENT_2', PerUnitStorageThroughput: 1000 }),
    });
    t.resourceCountIs('AWS::FSx::DataRepositoryAssociation', 1);
    t.hasResourceProperties('AWS::FSx::DataRepositoryAssociation', {
      S3: Match.objectLike({
        AutoImportPolicy: { Events: ['NEW', 'CHANGED', 'DELETED'] },
        AutoExportPolicy: { Events: ['NEW', 'CHANGED', 'DELETED'] },
      }),
    });
  });

  test('exposes the FSx mount outputs the lifecycle script needs', () => {
    t.hasOutput('*', { Description: Match.stringLikeRegexp('FSx Lustre filesystem id') });
    t.hasOutput('*', { Description: Match.stringLikeRegexp('FSx Lustre mount name') });
  });
});

describe('RlHyperPodStack', () => {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'RBase', { env: ENV, namePrefix: 'pai' });
  const stack = new RlHyperPodStack(app, 'TestRlHyperPod', { env: ENV, namePrefix: 'pai', base });
  const t = Template.fromStack(stack);

  test('synthesizes one HyperPod CfnCluster with the RL controller + worker groups', () => {
    t.resourceCountIs('AWS::SageMaker::Cluster', 1);
    t.hasResourceProperties('AWS::SageMaker::Cluster', {
      InstanceGroups: Match.arrayWith([
        Match.objectLike({ InstanceGroupName: 'controller-machine', InstanceType: 'ml.c5.2xlarge' }),
        Match.objectLike({ InstanceGroupName: 'worker-group-1', InstanceType: 'ml.g6.12xlarge', InstanceCount: 2 }),
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
      // A valid Slurm cluster needs a Controller group; keep it in the override example.
      instanceGroups: [
        { name: 'controller-machine', instanceType: 'ml.c5.2xlarge', instanceCount: 1, nodeType: 'Controller' },
        { name: 'worker', instanceType: 'ml.p5.48xlarge', instanceCount: 4, nodeType: 'Compute' },
      ],
    });
    const t2 = Template.fromStack(s2);
    t2.hasResourceProperties('AWS::SageMaker::Cluster', {
      InstanceGroups: Match.arrayWith([
        Match.objectLike({ InstanceType: 'ml.p5.48xlarge', InstanceCount: 4 }),
      ]),
    });
  });
});
