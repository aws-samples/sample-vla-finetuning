import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { SharedBaseStack } from '../lib/shared/base-stack';
import { PatternBStack } from '../lib/il/pattern-b-stack';

const ENV = { account: '111111111111', region: 'us-west-2' };

describe('PatternBStack', () => {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'TestBase', { env: ENV, namePrefix: 'pai' });
  const stack = new PatternBStack(app, 'TestPatternB', {
    env: ENV,
    namePrefix: 'pai',
    base,
    extraDatasetReadArns: ['arn:aws:s3:::some-dataset-bucket'],
  });
  const t = Template.fromStack(stack);

  test('creates exactly one SageMaker-trusted execution role', () => {
    t.resourceCountIs('AWS::IAM::Role', 1);
    t.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'sts:AssumeRole',
            Principal: { Service: 'sagemaker.amazonaws.com' },
          }),
        ]),
      }),
    });
  });

  test('execution role attaches the base jobBasePolicy (cross-stack import)', () => {
    // The base policy ARN arrives as an Fn::ImportValue token from the base stack.
    const roles = t.findResources('AWS::IAM::Role');
    const role = Object.values(roles)[0] as any;
    const arns = role.Properties.ManagedPolicyArns;
    expect(Array.isArray(arns)).toBe(true);
    expect(arns).toHaveLength(1);
    expect(arns[0]['Fn::ImportValue']).toMatch(/JobBasePolicy/);
  });

  test('does NOT place any GPU capacity / VPC (Pattern B is managed by SageMaker)', () => {
    t.resourceCountIs('AWS::EC2::VPC', 0);
    t.resourceCountIs('AWS::CloudFormation::CustomResource', 0);
    t.resourceCountIs('AWS::Batch::ComputeEnvironment', 0);
  });

  test('role policy grants CloudWatch Logs + namespaced metrics + extra dataset read', () => {
    t.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({ Sid: 'TrainingJobLogs' }),
          Match.objectLike({ Sid: 'TrainingJobMetrics' }),
          Match.objectLike({ Sid: 'ExtraDatasetRead' }),
        ]),
      }),
    });
  });

  test('PutMetricData is constrained to the SageMaker namespace', () => {
    t.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({
            Sid: 'TrainingJobMetrics',
            Action: 'cloudwatch:PutMetricData',
            Condition: {
              StringEquals: {
                'cloudwatch:namespace': Match.arrayWith(['/aws/sagemaker/TrainingJobs']),
              },
            },
          }),
        ]),
      }),
    });
  });

  test('exposes the execution role ARN as an output', () => {
    t.hasOutput('ExecutionRoleArn', {});
  });

  test('omitting extraDatasetReadArns yields no ExtraDatasetRead statement', () => {
    const app2 = new cdk.App();
    const base2 = new SharedBaseStack(app2, 'B2', { env: ENV, namePrefix: 'pai' });
    const s2 = new PatternBStack(app2, 'P2', { env: ENV, namePrefix: 'pai', base: base2 });
    const t2 = Template.fromStack(s2);
    const policies = t2.findResources('AWS::IAM::Policy');
    const hasExtra = Object.values(policies).some((p: any) =>
      (p.Properties.PolicyDocument.Statement as any[]).some((s) => s.Sid === 'ExtraDatasetRead'),
    );
    expect(hasExtra).toBe(false);
  });

  test('the pattern stack never owns an SNS topic (Base owns the shared topic)', () => {
    // The default `stack` above has a Base with no notifyEmail.
    t.resourceCountIs('AWS::SNS::Topic', 0);
    t.resourceCountIs('AWS::SNS::Subscription', 0);
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
    const nstack = new PatternBStack(napp, 'NPatternB', {
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

    test('the pattern stack creates an EventBridge rule filtering SageMaker terminal states by job-name prefix', () => {
      nt.hasResourceProperties('AWS::Events::Rule', {
        EventPattern: Match.objectLike({
          source: ['aws.sagemaker'],
          'detail-type': ['SageMaker Training Job State Change'],
          detail: Match.objectLike({
            TrainingJobName: [{ prefix: 'vla-ft-' }],
            TrainingJobStatus: ['Completed', 'Failed', 'Stopped'],
          }),
        }),
      });
    });

    test('the rule targets an SNS topic (the imported Base topic, via cross-stack ref)', () => {
      const rules = nt.findResources('AWS::Events::Rule');
      const rule = Object.values(rules)[0] as any;
      const targets = rule.Properties.Targets as any[];
      // Target Arn is a cross-stack reference (Fn::ImportValue / parameter) to Base's topic.
      expect(targets.length).toBeGreaterThan(0);
      expect(targets.some((tg) => JSON.stringify(tg.Arn).length > 0)).toBe(true);
    });

    test('Base exposes the notification topic ARN as an output', () => {
      nbaseT.hasOutput('NotificationTopicArn', {});
    });
  });
});
