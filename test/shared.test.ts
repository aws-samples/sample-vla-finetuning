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

  test('ECR repos use AES256 (the ECR default), NOT KMS — encryption config is omitted', () => {
    // Deliberately AES256, not KMS: EncryptionConfiguration is immutable, so flipping an
    // already-deployed AES256 repo to KMS forces a repo REPLACE. These repos are RETAIN +
    // custom-named, so CloudFormation cannot replace them and the Base update rolls back
    // (the landmine that blocked the SNS-notification fix; the repos hold verified images
    // such as the :efa multi-node fabric image). AES256 is the L2 default, which CDK emits
    // by omitting EncryptionConfiguration entirely — assert it is absent on all 3 repos.
    const repos = t.findResources('AWS::ECR::Repository');
    const withConfig = Object.values(repos).filter(
      (r: any) => r.Properties?.EncryptionConfiguration !== undefined,
    );
    expect(withConfig).toHaveLength(0);
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

describe('packPyBootstrap', () => {
  const fs = require('fs');
  const path = require('path');
  const { execFileSync } = require('child_process');
  const { stripPyCommentLines, packPyBootstrap } = require('../lib/shared/py-bootstrap');

  const bootstraps = [
    path.join(__dirname, '..', 'containers', 'isaac-lab-rl', 'rl_train_bootstrap.py'),
    path.join(__dirname, '..', 'containers', 'gr00t-n17', 'gr00t_train_bootstrap.py'),
  ];

  // Comments never reach the AST, so an identical ast.dump proves the strip changed
  // nothing the interpreter sees. A `#` line inside a multi-line string would change a
  // string constant and fail this.
  const AST_EQ =
    'import ast,sys\n' +
    'a,b=(open(p).read() for p in sys.argv[1:3])\n' +
    'sys.exit(0 if ast.dump(ast.parse(a))==ast.dump(ast.parse(b)) else 1)\n';

  test.each(bootstraps)('stripping comment lines keeps the AST of %s', (src: string) => {
    const tmp = path.join(require('os').tmpdir(), `stripped-${path.basename(src)}`);
    fs.writeFileSync(tmp, stripPyCommentLines(fs.readFileSync(src, 'utf8')));
    expect(() => execFileSync('python3', ['-c', AST_EQ, src, tmp])).not.toThrow();
  });

  test('a comment-looking line inside a string literal changes the AST (the guard bites)', () => {
    const src = 'X = """\n# not a comment\n"""\n';
    const os = require('os');
    const a = path.join(os.tmpdir(), 'guard-a.py');
    const b = path.join(os.tmpdir(), 'guard-b.py');
    fs.writeFileSync(a, src);
    fs.writeFileSync(b, stripPyCommentLines(src));
    expect(() => execFileSync('python3', ['-c', AST_EQ, a, b], { stdio: 'pipe' })).toThrow();
  });

  test('keeps a first-line shebang and drops indented comment lines', () => {
    const out = stripPyCommentLines('#!/usr/bin/env python3\n# a\nx = 1  # inline kept\n    # b\ny = 2\n');
    expect(out).toBe('#!/usr/bin/env python3\nx = 1  # inline kept\ny = 2\n');
  });

  test('the packed stub is smaller than packing the raw source', () => {
    const zlib = require('zlib');
    const raw = fs.readFileSync(bootstraps[0], 'utf8');
    const rawPacked = zlib.deflateSync(Buffer.from(raw, 'utf8'), { level: 9 }).toString('base64');
    expect(packPyBootstrap(raw, 'x.py').length).toBeLessThan(rawPacked.length);
  });
});
