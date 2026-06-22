import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { SharedBaseStack } from '../lib/shared/base-stack';
import {
  AzSelectorConstruct,
  DEFAULT_INSTANCE_TYPE_FALLBACK,
} from '../lib/shared/az-selector';

const ENV = { account: '111111111111', region: 'us-west-2' };

describe('SharedBaseStack', () => {
  const app = new cdk.App();
  const stack = new SharedBaseStack(app, 'TestBase', { env: ENV, namePrefix: 'pai' });
  const t = Template.fromStack(stack);

  test('creates one VPC with a NAT gateway and an S3 gateway endpoint', () => {
    t.resourceCountIs('AWS::EC2::VPC', 1);
    t.resourceCountIs('AWS::EC2::NatGateway', 1);
    t.hasResourceProperties('AWS::EC2::VPCEndpoint', {
      VpcEndpointType: 'Gateway',
    });
  });

  test('three RETAIN, encrypted, versioned, public-blocked buckets (data, artifacts, access-logs)', () => {
    // data + artifacts + the dedicated server-access-log destination.
    t.resourceCountIs('AWS::S3::Bucket', 3);
    // Every bucket is RETAIN.
    const buckets = t.findResources('AWS::S3::Bucket');
    Object.values(buckets).forEach((b: any) => {
      expect(b.DeletionPolicy).toBe('Retain');
      expect(b.Properties.VersioningConfiguration.Status).toBe('Enabled');
      expect(b.Properties.PublicAccessBlockConfiguration.BlockPublicAcls).toBe(true);
    });
  });

  test('data + artifact buckets ship access logs to the dedicated log bucket', () => {
    // Two buckets carry a LoggingConfiguration (the log bucket itself does not, by design).
    const buckets = t.findResources('AWS::S3::Bucket');
    const logged = Object.values(buckets).filter(
      (b: any) => b.Properties.LoggingConfiguration !== undefined,
    );
    expect(logged.length).toBe(2);
  });

  test('ECR repos are KMS-encrypted', () => {
    t.hasResourceProperties('AWS::ECR::Repository', {
      EncryptionConfiguration: { EncryptionType: 'KMS' },
    });
  });

  test('three ECR repos with scan-on-push (vla-ft, isaac-lab-rl, gr00t-n17)', () => {
    t.resourceCountIs('AWS::ECR::Repository', 3);
    t.hasResourceProperties('AWS::ECR::Repository', {
      ImageScanningConfiguration: { ScanOnPush: true },
    });
  });

  test('EFS file system is encrypted and RETAIN', () => {
    t.hasResource('AWS::EFS::FileSystem', {
      DeletionPolicy: 'Retain',
      Properties: { Encrypted: true },
    });
  });

  test('base job policy grants S3 + ECR access', () => {
    // Match.arrayWith matches an ordered subsequence, so list sids in document order.
    t.hasResourceProperties('AWS::IAM::ManagedPolicy', {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({ Sid: 'ReadWriteArtifactBucket' }),
          Match.objectLike({ Sid: 'PullContainers' }),
        ]),
      }),
    });
  });

  test('base stack synthesizes with no AzSelector custom resource', () => {
    // AzSelector is opt-in; base stack must not place GPU-probe Lambdas.
    t.resourceCountIs('AWS::CloudFormation::CustomResource', 0);
  });
});

describe('AzSelectorConstruct', () => {
  test('default fallback prioritizes g6e then g5', () => {
    expect(DEFAULT_INSTANCE_TYPE_FALLBACK[0]).toMatch(/^g6e\./);
    expect(DEFAULT_INSTANCE_TYPE_FALLBACK).toContain('g5.4xlarge');
  });

  test('emits a Lambda-backed custom resource with EC2 probe permissions', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'AzTest', { env: ENV });
    new AzSelectorConstruct(stack, 'Az', {
      instanceTypes: DEFAULT_INSTANCE_TYPE_FALLBACK,
      amiId: 'ami-0123456789abcdef0',
    });
    const t = Template.fromStack(stack);

    t.resourceCountIs('AWS::CloudFormation::CustomResource', 1);
    t.hasResourceProperties('AWS::Lambda::Function', {
      Runtime: 'python3.12',
      Handler: 'index.handler',
    });
    // The probe role can launch/terminate instances and list offerings.
    t.hasResourceProperties('AWS::IAM::Role', {
      Policies: Match.arrayWith([
        Match.objectLike({
          PolicyDocument: {
            Statement: Match.arrayWith([
              Match.objectLike({
                Action: Match.arrayWith([
                  'ec2:DescribeInstanceTypeOfferings',
                  'ec2:RunInstances',
                  'ec2:TerminateInstances',
                ]),
              }),
            ]),
          },
        }),
      ]),
    });
  });
});
