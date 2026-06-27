/**
 * FsxLustreFileSystem — FSx for Lustre hot tier + S3 Data Repository Association (DRA).
 *
 * The multi-node data plane for Pattern C: large LeRobot datasets + DCP checkpoints live
 * in an in-region S3 origin (the platform dataBucket / artifactBucket) and hydrate
 * on-demand onto an FSx-Lustre filesystem that every node mounts over EFA. This replaces
 * the single-node pattern of `aws s3 cp`-ing a dataset onto one box — at multi-node scale
 * a Lustre+DRA link gives a POSIX hot tier shared by all ranks with lazy S3 hydration and
 * automatic write-back, with NO manual staging step.
 *
 * Absorbed from awslabs/awsome-distributed-ai Cosmos3 (MIT-0, storage-fsx-efa-sc.yaml +
 * storage-fsx-dra.yaml, fetched 2026-06-27). The verified knobs are copied as-is:
 *   - deploymentType PERSISTENT_2, perUnitStorageThroughput 1000 MBps/TiB (the AWS GDS
 *     reference tier), fileSystemTypeVersion 2.15, storageCapacity 9600 GiB.
 *   - A DRA (NOT the L2's legacy importPath/exportPath, which can't auto-EXPORT) with BOTH
 *     autoImportPolicy AND autoExportPolicy = NEW|CHANGED|DELETED — the exact policy the
 *     Cosmos3 `aws fsx create-data-repository-association --s3 'AutoImportPolicy={Events=
 *     [NEW,CHANGED,DELETED]},AutoExportPolicy={Events=[NEW,CHANGED,DELETED]}'` sets.
 *
 * Why a separate CfnDataRepositoryAssociation rather than the L2 LustreConfiguration
 * importPath/exportPath: that legacy single-link only auto-imports; bidirectional sync with
 * a DELETED policy (so a checkpoint pruned on Lustre is also pruned in S3) needs a DRA.
 *
 * COST GATE: a PERSISTENT_2 filesystem bills continuously (provisioned throughput × TiB)
 * the moment it exists — unlike the Batch CEs which are $0 idle. So this construct is NOT
 * instantiated by the base or any pattern stack that deploys by default; it is created only
 * by an explicitly-gated multi-node deploy (Phase 4) or a standalone FsxLustreStack an
 * operator deploys when a multi-node job needs it, then tears down after.
 *
 * TEARDOWN: removalPolicy defaults to DESTROY because the Lustre FS is a CACHE — the durable
 * copy lives in S3 (DRA exports writes back). Unlike a RETAIN FSx ONTAP vault (whose
 * non-root-volume teardown order is a known trap), a single Lustre filesystem with a DRA
 * tears down cleanly: CFN deletes the DRA first (it depends on the FS), then the FS. Set
 * RETAIN only if you want the hydrated cache to survive a stack delete (rare — S3 already has it).
 */
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as fsx from 'aws-cdk-lib/aws-fsx';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface FsxLustreFileSystemProps {
  /** VPC the filesystem lives in (the platform VPC). */
  readonly vpc: ec2.IVpc;
  /**
   * The S3 bucket the DRA links to (datasets + checkpoints hydrate from / export to here).
   * Typically the platform dataBucket or artifactBucket.
   */
  readonly dataRepositoryBucket: s3.IBucket;
  /**
   * S3 prefix under the bucket the DRA links (e.g. 'fsx-datasets/'). The Lustre path
   * `fileSystemPath` mirrors it. Default '' (the whole bucket) — set a prefix to scope the link.
   */
  readonly dataRepositoryPrefix?: string;
  /**
   * Lustre mount path the DRA links the S3 path to. Cosmos3 uses '/datasets'. Must start
   * with '/'. Default '/datasets'.
   */
  readonly fileSystemPath?: string;
  /**
   * Storage capacity in GiB. PERSISTENT_2 valid values: 1200, 2400, then multiples of 2400.
   * Default 9600 (Cosmos3's sizing: 9.6 TiB @ 1000 MBps/TiB for dataset + DCP headroom).
   */
  readonly storageCapacityGiB?: number;
  /**
   * Per-unit storage throughput (MBps per TiB). PERSISTENT_2 valid: 125/250/500/1000.
   * Default 1000 (the AWS GDS reference tier Cosmos3 uses).
   */
  readonly perUnitStorageThroughput?: number;
  /**
   * The subnet to place the (single-AZ) filesystem in. Lustre is single-subnet; default is
   * the first PRIVATE_WITH_EGRESS subnet of the VPC. Pin it to the AZ your GPU nodes land in.
   */
  readonly vpcSubnet?: ec2.ISubnet;
  /**
   * RemovalPolicy for the filesystem. Default DESTROY (the FS is an S3-backed cache).
   * Set RETAIN to keep the hydrated cache across a stack delete.
   */
  readonly removalPolicy?: cdk.RemovalPolicy;
}

export class FsxLustreFileSystem extends Construct {
  public readonly fileSystem: fsx.LustreFileSystem;
  /** Self-referencing all-traffic SG (EFA + Lustre ports 988/1018-1023). Mount clients join this. */
  public readonly securityGroup: ec2.SecurityGroup;
  public readonly dataRepositoryAssociation: fsx.CfnDataRepositoryAssociation;
  /** Lustre mount path the S3 repository is linked at (e.g. /datasets). */
  public readonly fileSystemPath: string;
  /** The S3 path the DRA links (s3://bucket/prefix/). */
  public readonly dataRepositoryPath!: string;

  constructor(scope: Construct, id: string, props: FsxLustreFileSystemProps) {
    super(scope, id);

    const fileSystemPath = props.fileSystemPath ?? '/datasets';
    if (!fileSystemPath.startsWith('/')) {
      throw new Error(`fileSystemPath must start with '/', got '${fileSystemPath}'`);
    }
    this.fileSystemPath = fileSystemPath;
    const prefix = props.dataRepositoryPrefix ?? '';
    const dataRepositoryPath = `s3://${props.dataRepositoryBucket.bucketName}/${prefix}`;

    // SG: self-referencing all-traffic (mirrors the Cosmos3 EFA security group + covers the
    // Lustre client ports 988 and 1018-1023 in one rule). EFA-backed Lustre needs the EFA
    // all-traffic rule anyway, and GPU nodes join this SG to mount.
    this.securityGroup = new ec2.SecurityGroup(this, 'FsxSg', {
      vpc: props.vpc,
      allowAllOutbound: true,
      description: 'FSx Lustre + EFA (self-referencing all-traffic; mount clients join this SG)',
    });
    this.securityGroup.connections.allowInternally(
      ec2.Port.allTraffic(),
      'FSx Lustre client ports (988, 1018-1023) + EFA RDMA between mount clients',
    );

    const vpcSubnet =
      props.vpcSubnet ??
      props.vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }).subnets[0];

    // The Lustre filesystem. Deliberately NO importPath/exportPath here: the bidirectional
    // sync (incl. DELETED) is done by the DRA below, which the L2 legacy single-link cannot
    // express. PERSISTENT_2 defaults its metadata configuration to AUTOMATIC (what the
    // Cosmos3 YAML sets explicitly), so we don't need an escape hatch for it.
    this.fileSystem = new fsx.LustreFileSystem(this, 'FileSystem', {
      vpc: props.vpc,
      vpcSubnet,
      securityGroup: this.securityGroup,
      storageCapacityGiB: props.storageCapacityGiB ?? 9600,
      fileSystemTypeVersion: fsx.FileSystemTypeVersion.V_2_15,
      lustreConfiguration: {
        deploymentType: fsx.LustreDeploymentType.PERSISTENT_2,
        perUnitStorageThroughput: props.perUnitStorageThroughput ?? 1000,
        // Raw I/O tier — benchmark/throughput-bound, not compression-bound (Cosmos3 NONE).
        dataCompressionType: fsx.LustreDataCompressionType.NONE,
      },
    });
    this.fileSystem.applyRemovalPolicy(props.removalPolicy ?? cdk.RemovalPolicy.DESTROY);

    // The DRA: link S3 <-> the Lustre path with bidirectional auto-sync. autoImportPolicy
    // hydrates NEW/CHANGED objects + propagates DELETED; autoExportPolicy writes Lustre
    // changes back to S3 the same way. batchImportMetaDataOnCreate pre-loads the existing S3
    // object metadata at create so the first read of an existing dataset doesn't stall. This
    // is the verbatim Cosmos3 policy (Events=[NEW,CHANGED,DELETED] both directions).
    this.dataRepositoryAssociation = new fsx.CfnDataRepositoryAssociation(this, 'Dra', {
      fileSystemId: this.fileSystem.fileSystemId,
      fileSystemPath,
      dataRepositoryPath,
      batchImportMetaDataOnCreate: true,
      s3: {
        autoImportPolicy: { events: ['NEW', 'CHANGED', 'DELETED'] },
        autoExportPolicy: { events: ['NEW', 'CHANGED', 'DELETED'] },
      },
    });

    // The S3 path the DRA links — exposed for the owning stack's outputs (we deliberately
    // do NOT create CfnOutputs in the construct: a reusable construct's outputs get a
    // path-prefixed logical id and would collide if instantiated twice. The owning
    // stack/test wires CfnOutputs from these public properties + the L2's fileSystemId /
    // mountName / dnsName).
    this.dataRepositoryPath = dataRepositoryPath;
  }
}
