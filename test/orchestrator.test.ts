import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { SharedBaseStack } from '../lib/shared/base-stack';
import { PatternAStack } from '../lib/il/pattern-a-stack';
import { PatternBStack } from '../lib/il/pattern-b-stack';
import { GrootPatternAStack } from '../lib/il/gr00t-pattern-a-stack';
import { RlPatternAStack } from '../lib/rl/pattern-a-stack';
import { OrchestratorStack } from '../lib/orchestrator/orchestrator-stack';

const ENV = { account: '111111111111', region: 'us-west-2' };

// The orchestrator is deploy-gated (NOT in bin/app.ts, like the HyperPod stacks), so
// this synth IS the gate: it proves the Step Functions state machine + the two Lambdas
// wire up against the real pattern stacks. Cost 0.
function build() {
  const app = new cdk.App();
  const base = new SharedBaseStack(app, 'TestBase', { env: ENV, namePrefix: 'pai' });
  const ilA = new PatternAStack(app, 'TestIlA', { env: ENV, namePrefix: 'pai', base });
  const ilB = new PatternBStack(app, 'TestIlB', { env: ENV, namePrefix: 'pai', base });
  const rlA = new RlPatternAStack(app, 'TestRlA', { env: ENV, namePrefix: 'pai', base });
  const grootA = new GrootPatternAStack(app, 'TestGrootA', { env: ENV, namePrefix: 'pai', base });
  const orch = new OrchestratorStack(app, 'TestOrchestrator', {
    env: ENV,
    namePrefix: 'pai',
    base,
    ilPatternA: ilA,
    ilPatternB: ilB,
    rlPatternA: rlA,
    grootPatternA: grootA,
    notifyEmail: 'you@example.com',
  });
  return Template.fromStack(orch);
}

describe('OrchestratorStack', () => {
  const t = build();

  test('synthesizes a Step Functions state machine', () => {
    t.resourceCountIs('AWS::StepFunctions::StateMachine', 1);
  });

  test('two Python 3.12 Lambdas: plan + submit', () => {
    const fns = t.findResources('AWS::Lambda::Function');
    const handlers = Object.values(fns).map((f: any) => f.Properties.Handler);
    expect(handlers).toContain('orchestrator_plan.handler');
    expect(handlers).toContain('orchestrator_submit.handler');
    Object.values(fns).forEach((f: any) => {
      expect(f.Properties.Runtime).toBe('python3.12');
    });
  });

  test('the state machine definition is deterministic — no Bedrock/LLM task in the path', () => {
    const machines = t.findResources('AWS::StepFunctions::StateMachine');
    const def = JSON.stringify(Object.values(machines)[0]);
    expect(def.toLowerCase()).not.toContain('bedrock');
    // The two decision/dispatch steps + the runnable Choice + the recommend fallback.
    expect(def).toContain('Runnable?');
    expect(def).toContain('RecommendOnly');
  });

  test('the submit Lambda is wired (env) to ALL THREE Batch backends + the SageMaker handoff', () => {
    const fns = t.findResources('AWS::Lambda::Function');
    const submit = Object.values(fns).find(
      (f: any) => f.Properties.Handler === 'orchestrator_submit.handler',
    ) as any;
    const envVars = submit.Properties.Environment.Variables;
    // IL Pattern A (lerobot Batch) + RL Pattern A (Batch) + GR00T Pattern A (Batch) +
    // Pattern B (SageMaker handoff).
    for (const k of [
      'IL_A_JOB_QUEUE',
      'IL_A_JOB_DEFINITION',
      'IL_A_CODE_S3',
      'IL_A_OUTPUT_S3',
      'RL_A_JOB_QUEUE',
      'RL_A_JOB_DEFINITION',
      'RL_A_OUTPUT_S3',
      'GROOT_A_JOB_QUEUE',
      'GROOT_A_JOB_DEFINITION',
      'GROOT_A_OUTPUT_S3',
      'B_EXECUTION_ROLE',
      'B_IMAGE_URI',
      'B_OUTPUT_S3',
      'HF_TOKEN_SSM',
      'HF_TOKEN_SSM_REGION',
    ]) {
      expect(envVars[k]).toBeDefined();
    }
  });

  test('submit Lambda role can SubmitJob on BOTH platform queues (least-privilege, not *)', () => {
    const policies = t.findResources('AWS::IAM::Policy');
    const submitStmts = Object.values(policies).flatMap((p: any) =>
      (p.Properties.PolicyDocument.Statement as any[]).filter((s) => s.Sid === 'SubmitBatchJobs'),
    );
    expect(submitStmts.length).toBe(1);
    const stmt = submitStmts[0];
    expect(stmt.Action).toBe('batch:SubmitJob');
    // Resources are the imported queue/jobdef ARNs (cross-stack refs), NOT '*'.
    expect(JSON.stringify(stmt.Resource)).not.toContain('"*"');
  });

  test('submit Lambda role can read the HF token from SSM (scoped to the parameter)', () => {
    const policies = t.findResources('AWS::IAM::Policy');
    const ssmStmts = Object.values(policies).flatMap((p: any) =>
      (p.Properties.PolicyDocument.Statement as any[]).filter((s) => s.Sid === 'ReadHfToken'),
    );
    expect(ssmStmts.length).toBe(1);
    expect(ssmStmts[0].Action).toBe('ssm:GetParameter');
    expect(JSON.stringify(ssmStmts[0].Resource)).toContain(':parameter/pai/hf-token');
  });

  test('a Choice state routes on the plan-determined runnable flag (Pattern C → recommend)', () => {
    const machines = t.findResources('AWS::StepFunctions::StateMachine');
    const def = JSON.stringify(Object.values(machines)[0]);
    expect(def).toContain('$.runnable');
  });

  test('outcome + failure notifications publish to one SNS topic', () => {
    t.resourceCountIs('AWS::SNS::Topic', 1);
    // The state machine references SnsPublish targets; the email subscription is created.
    t.hasResourceProperties('AWS::SNS::Subscription', {
      Protocol: 'email',
      Endpoint: 'you@example.com',
    });
  });

  test('exposes the state machine ARN as an output', () => {
    t.hasOutput('StateMachineArn', {});
  });

  test('IAM role descriptions / statements are ASCII-only (no em-dash → deploy 400)', () => {
    // An em-dash in an IAM description synths fine but 400s on deploy.
    // Guard the whole synthesized template.
    const json = JSON.stringify(t.toJSON());
    // eslint-disable-next-line no-control-regex
    const nonAscii = json.match(/[^\x00-\x7F]/g);
    expect(nonAscii).toBeNull();
  });
});
