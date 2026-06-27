import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';
import { FsxLustreFileSystem } from '../lib/shared/fsx-lustre';

const ENV = { account: '111111111111', region: 'us-west-2' };

// Thin host stack: a VPC + bucket + the construct under test. Mirrors how a gated Phase 4
// (or a standalone FsxLustreStack) would instantiate it.
class HostStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps & { storageCapacityGiB?: number }) {
    super(scope, id, props);
    const vpc = new ec2.Vpc(this, 'Vpc', { maxAzs: 2, natGateways: 1 });
    const bucket = new s3.Bucket(this, 'DataBucket');
    const fsx = new FsxLustreFileSystem(this, 'Fsx', {
      vpc,
      dataRepositoryBucket: bucket,
      dataRepositoryPrefix: 'fsx-datasets/',
      storageCapacityGiB: props?.storageCapacityGiB,
    });
    // The owning stack wires the outputs from the construct's public properties (the
    // construct itself does not, to stay reusable). This is exactly the Phase 4 pattern.
    new cdk.CfnOutput(this, 'FsxFileSystemId', { value: fsx.fileSystem.fileSystemId });
    new cdk.CfnOutput(this, 'FsxMountName', { value: fsx.fileSystem.mountName });
    new cdk.CfnOutput(this, 'FsxDataRepositoryPath', { value: fsx.dataRepositoryPath });
  }
}

describe('FsxLustreFileSystem', () => {
  const app = new cdk.App();
  const stack = new HostStack(app, 'TestFsx', { env: ENV });
  const t = Template.fromStack(stack);

  test('creates ONE PERSISTENT_2 Lustre FS at 1000 MBps/TiB, FS version 2.15, 9600 GiB default', () => {
    t.resourceCountIs('AWS::FSx::FileSystem', 1);
    t.hasResourceProperties('AWS::FSx::FileSystem', {
      FileSystemType: 'LUSTRE',
      FileSystemTypeVersion: '2.15',
      StorageCapacity: 9600,
      LustreConfiguration: Match.objectLike({
        DeploymentType: 'PERSISTENT_2',
        PerUnitStorageThroughput: 1000,
        DataCompressionType: 'NONE',
      }),
    });
  });

  test('does NOT use the legacy single-link import/export on the FS (the DRA does the sync)', () => {
    // ImportPath/ExportPath/AutoImportPolicy would be the legacy one-way link; we use a DRA
    // instead so auto-EXPORT + DELETED works. Assert they are absent from the FS config.
    const fss = t.findResources('AWS::FSx::FileSystem');
    const cfg = (Object.values(fss)[0] as any).Properties.LustreConfiguration;
    expect(cfg.ImportPath).toBeUndefined();
    expect(cfg.ExportPath).toBeUndefined();
    expect(cfg.AutoImportPolicy).toBeUndefined();
  });

  test('creates a DRA with BIDIRECTIONAL NEW/CHANGED/DELETED auto import + export', () => {
    t.resourceCountIs('AWS::FSx::DataRepositoryAssociation', 1);
    t.hasResourceProperties('AWS::FSx::DataRepositoryAssociation', {
      FileSystemPath: '/datasets',
      BatchImportMetaDataOnCreate: true,
      S3: Match.objectLike({
        AutoImportPolicy: { Events: ['NEW', 'CHANGED', 'DELETED'] },
        AutoExportPolicy: { Events: ['NEW', 'CHANGED', 'DELETED'] },
      }),
    });
  });

  test('the DRA links the bucket + prefix as the S3 data repository path', () => {
    const dras = t.findResources('AWS::FSx::DataRepositoryAssociation');
    const dra = Object.values(dras)[0] as any;
    // DataRepositoryPath is s3://<bucket>/fsx-datasets/ — bucket name is a token (Ref/Join),
    // but the prefix is a literal we can assert is present in the joined value.
    const path = JSON.stringify(dra.Properties.DataRepositoryPath);
    expect(path).toContain('fsx-datasets/');
    expect(path).toContain('s3://');
  });

  test('self-referencing all-traffic SG (Lustre client ports + EFA between mount clients)', () => {
    // An ingress rule whose source is the SG itself (self-ref) covering all traffic.
    t.hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
      IpProtocol: '-1',
    });
  });

  test('default removalPolicy is DESTROY (the FS is an S3-backed cache; S3 has the durable copy)', () => {
    const fss = t.findResources('AWS::FSx::FileSystem');
    const fsRes = Object.values(fss)[0] as any;
    expect(fsRes.DeletionPolicy).toBe('Delete');
  });

  test('the owning stack can wire outputs from the construct public properties', () => {
    t.hasOutput('FsxFileSystemId', {});
    t.hasOutput('FsxMountName', {});
    t.hasOutput('FsxDataRepositoryPath', {});
  });

  test('RETAIN removalPolicy is honored when requested', () => {
    const app2 = new cdk.App();
    class RetainHost extends cdk.Stack {
      constructor(scope: Construct, id: string) {
        super(scope, id, { env: ENV });
        const vpc = new ec2.Vpc(this, 'Vpc', { maxAzs: 2, natGateways: 1 });
        const bucket = new s3.Bucket(this, 'B');
        new FsxLustreFileSystem(this, 'Fsx', {
          vpc,
          dataRepositoryBucket: bucket,
          removalPolicy: cdk.RemovalPolicy.RETAIN,
        });
      }
    }
    const t2 = Template.fromStack(new RetainHost(app2, 'RetainFsx'));
    const fsRes = Object.values(t2.findResources('AWS::FSx::FileSystem'))[0] as any;
    expect(fsRes.DeletionPolicy).toBe('Retain');
  });

  test('rejects a fileSystemPath that does not start with "/"', () => {
    const app3 = new cdk.App();
    class BadHost extends cdk.Stack {
      constructor(scope: Construct, id: string) {
        super(scope, id, { env: ENV });
        const vpc = new ec2.Vpc(this, 'Vpc', { maxAzs: 2, natGateways: 1 });
        const bucket = new s3.Bucket(this, 'B');
        new FsxLustreFileSystem(this, 'Fsx', {
          vpc,
          dataRepositoryBucket: bucket,
          fileSystemPath: 'datasets', // missing leading slash
        });
      }
    }
    expect(() => new BadHost(app3, 'BadFsx')).toThrow(/must start with/);
  });
});
