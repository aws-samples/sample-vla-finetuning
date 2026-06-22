/**
 * AzSelectorConstruct
 *
 * A deploy-time Custom Resource Lambda that probes for an Availability Zone +
 * instance type with actual GPU capacity, by attempting a throwaway RunInstances
 * down a fallback list. Solves the recurring "Spot/On-Demand InsufficientInstance
 * Capacity" failure that both axes (IL fine-tune, RL training) hit on g-series GPUs.
 *
 * Extracted from a verified Isaac Lab `AzSelector` construct (in production use) and
 * generalized: the default fallback list now
 * reflects the platform's secured quota reality (g6e L40S → g5 A10G), and the
 * probe AMI is a prop so callers pass a region-correct GPU AMI.
 *
 * Behaviour:
 *  1. Walk the instance-type fallback list in priority order.
 *  2. For each type, list supported AZs via describe-instance-type-offerings.
 *  3. Shuffle AZs (avoid hammering one AZ across deploys).
 *  4. RunInstances (MinCount=1) in each AZ; on success, terminate the probe and
 *     return that AZ + type.
 *  5. On InsufficientInstanceCapacity, try the next AZ; when a type is exhausted,
 *     fall back to the next type. If everything fails, the resource fails (so the
 *     stack does not silently deploy into a zone with no capacity).
 *
 * NOTE: this is opt-in. The shared base stack does NOT instantiate it by default,
 * so `cdk synth` of the base stack needs no credentials and no probe AMI. Pattern
 * stacks (A/B/RL) wire it in when they actually place GPU capacity.
 */
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';

export interface AzSelectorProps {
  /** Instance-type fallback list, highest priority first. */
  readonly instanceTypes: string[];
  /** AMI ID used for the throwaway capacity probe (must exist in the target region). */
  readonly amiId: string;
}

/**
 * Default fallback order for the platform, reflecting secured us-west-2 quota
 * (2026-06-14): g6e (L40S) preferred for training, g5 (A10G) as the fallback.
 * Single-GPU types last so the probe degrades to "something runnable" rather than
 * failing outright. Callers should override per workload (e.g. a 4-GPU job passes
 * only multi-GPU types).
 */
export const DEFAULT_INSTANCE_TYPE_FALLBACK = [
  'g6e.12xlarge', // L40S × 4 — distributed / large single-node FT
  'g6e.4xlarge', // L40S × 1 — single-GPU FT (primary, quota=4)
  'g5.12xlarge', // A10G × 4 — fallback multi-GPU
  'g5.4xlarge', // A10G × 1 — last resort
];

/**
 * Probes for an AZ + instance type with GPU capacity at deploy time.
 */
export class AzSelectorConstruct extends Construct {
  /** Resolved Availability Zone (CloudFormation runtime token). */
  public readonly availabilityZone: string;
  /** Resolved instance type (CloudFormation runtime token). */
  public readonly resolvedInstanceType: string;

  constructor(scope: Construct, id: string, props: AzSelectorProps) {
    super(scope, id);

    const lambdaRole = new iam.Role(this, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/AWSLambdaBasicExecutionRole',
        ),
      ],
      inlinePolicies: {
        AzSelectorPolicy: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: [
                'ec2:DescribeInstanceTypeOfferings',
                'ec2:RunInstances',
                'ec2:TerminateInstances',
                'ec2:DescribeInstances',
                'ec2:CreateTags',
              ],
              resources: ['*'],
            }),
          ],
        }),
      },
    });

    const fn = new lambda.Function(this, 'Function', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      role: lambdaRole,
      timeout: cdk.Duration.minutes(10),
      code: lambda.Code.fromInline(AZ_SELECTOR_LAMBDA_CODE),
      description:
        'Finds an AZ + instance type with GPU capacity via trial RunInstances with fallback',
    });

    const cr = new cdk.CustomResource(this, 'Resource', {
      serviceToken: fn.functionArn,
      properties: {
        InstanceTypes: props.instanceTypes.join(','),
        AmiId: props.amiId,
      },
    });

    fn.addPermission('CfnInvoke', {
      principal: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
    });

    this.availabilityZone = cr.getAttString('AvailabilityZone');
    this.resolvedInstanceType = cr.getAttString('InstanceType');
  }
}

/** AZ + instance-type capacity probe (Python, inline Lambda). */
const AZ_SELECTOR_LAMBDA_CODE = `
import json
import boto3
import random
import cfnresponse

def handler(event, context):
    print(json.dumps(event))

    if event['RequestType'] == 'Delete':
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {},
            physicalResourceId=event.get('PhysicalResourceId', 'az-selector-deleted'))
        return

    try:
        instance_types = event['ResourceProperties']['InstanceTypes'].split(',')
        ami_id = event['ResourceProperties']['AmiId']
        region = context.invoked_function_arn.split(':')[3]

        ec2 = boto3.client('ec2', region_name=region)
        all_tried = []

        for instance_type in instance_types:
            print(f'--- Trying instance type: {instance_type} ---')

            resp = ec2.describe_instance_type_offerings(
                LocationType='availability-zone',
                Filters=[{'Name': 'instance-type', 'Values': [instance_type]}]
            )
            azs = [o['Location'] for o in resp['InstanceTypeOfferings']]
            print(f'Supported AZs for {instance_type}: {azs}')

            if not azs:
                print(f'{instance_type} is not available in any AZ, skipping...')
                all_tried.append(f'{instance_type}(no AZ)')
                continue

            random.shuffle(azs)

            for az in azs:
                print(f'Trying {instance_type} in {az}')
                try:
                    run_resp = ec2.run_instances(
                        InstanceType=instance_type,
                        ImageId=ami_id,
                        MinCount=1,
                        MaxCount=1,
                        Placement={'AvailabilityZone': az},
                        TagSpecifications=[{
                            'ResourceType': 'instance',
                            'Tags': [{'Key': 'Name', 'Value': 'pai-az-selector-probe'}]
                        }]
                    )
                    instance_id = run_resp['Instances'][0]['InstanceId']
                    print(f'SUCCESS: {instance_type} in {az} (probe: {instance_id})')

                    ec2.terminate_instances(InstanceIds=[instance_id])
                    print(f'Terminated probe {instance_id}')

                    cfnresponse.send(event, context, cfnresponse.SUCCESS,
                        {'AvailabilityZone': az, 'InstanceType': instance_type},
                        physicalResourceId=f'az-selector-{az}-{instance_type}')
                    return

                except Exception as e:
                    error_msg = str(e)
                    if 'InsufficientInstanceCapacity' in error_msg:
                        print(f'InsufficientCapacity: {instance_type} in {az}')
                        all_tried.append(f'{instance_type}/{az}')
                        continue
                    elif 'Unsupported' in error_msg:
                        print(f'Unsupported: {instance_type} in {az}')
                        all_tried.append(f'{instance_type}/{az}(unsupported)')
                        continue
                    else:
                        print(f'Unexpected error: {e}')
                        raise

            print(f'All AZs exhausted for {instance_type}, falling back...')

        cfnresponse.send(event, context, cfnresponse.FAILED, {},
            reason=f'No capacity available for any instance type in any AZ: {all_tried}',
            physicalResourceId='az-selector-failed')

    except Exception as e:
        print(f'Error: {e}')
        cfnresponse.send(event, context, cfnresponse.FAILED, {},
            reason=str(e),
            physicalResourceId='az-selector-error')
`;
