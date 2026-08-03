# Parameter Reference — `pcs-ml-cluster-deploy-all.yaml`

Full parameter list for the all-in-one deployment template. The sections and their order
match the CloudFormation console's parameter groups exactly (the console shows friendly
labels via `AWS::CloudFormation::Interface`). Defaults give the most common production
setup — the latest PCS-Ready Deep Learning AMI auto-resolved from SSM, Enroot/Pyxis
installed at first boot via `PostInstallScriptUrl`, monitoring enabled — so a default
deploy only needs the Availability Zone (`PrimarySubnetAZ`). To pre-bake Enroot/Pyxis into
a custom AMI for faster boots, build it separately with
[`pcs-ready-dlami-with-enroot-pyxis.yaml`](../README.md#85-pre-baking-enrootpyxis-into-a-custom-ami)
and pass its output as `AmiId`.

For conceptual guidance (GPU instance/EFA selection, FSx Region availability, container
runtime options), see the [README](../README.md#4-configuration).

## 1. Network Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `PrimarySubnetAZ` | *(required)* | Availability Zone to deploy into — the one required parameter. Holds the public subnet (login node), the primary private subnet (compute, FSx), and the single NAT gateway |
| `AdditionalSubnetAZ2` | *(empty)* | (Optional) 2nd AZ for an additional private subnet. Empty = single-AZ. Enables multi-AZ layouts (e.g. OpenZFS `MULTI_AZ`). Shares the primary AZ's NAT gateway (cross-AZ egress, no per-AZ NAT) |
| `AdditionalSubnetAZ3` | *(empty)* | (Optional) 3rd AZ for an additional private subnet. Requires `AdditionalSubnetAZ2` to also be set. Max 3 private AZs total |
| `CreateS3Endpoint` | `true` | Create an S3 VPC endpoint |

The VPC name is fixed to `${StackName}-VPC` (derived from the stack name, so
multiple deployments in one account get unique VPC names automatically) — there
is no `VPCName` parameter on the all-in-one template. The standalone
`ml-cluster-prerequisites.yaml` still accepts a `VPCName` parameter if you deploy
it directly.

## 2. PCS Cluster Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `SlurmVersion` | `25.11` | Slurm version (`25.05` or `25.11`). Drives which monitoring you get (Slurm OpenMetrics is 25.11+ only) and is also threaded into the CNG UserData so the right-version Pyxis is installed; see [OPERATIONS.md §1](./OPERATIONS.md#1-slurm-version-selection) |
| `LoginNodeInstanceType` | `m6i.4xlarge` | Login node instance type |
| `RootVolumeSize` | `300` | Root EBS volume size (GiB) on every node (login + compute); 300 leaves room for large container images (Megatron `.sqsh` ~20 GB) |
| `AmiId` | *(empty → SSM auto-resolve)* | AMI ID for every node group. **Empty (default) auto-resolves to the latest PCS-Ready Deep Learning AMI** (Ubuntu 24.04, x86_64) from SSM (`/aws/service/pcs/ami/dlami-base-ubuntu2404/x86_64/latest/ami-id`). For production, **pin to a specific `ami-xxx`** so a later scale-out cannot drift onto a newer base. Use a custom AMI built off the PCS-Ready DLAMI base (e.g. via [`pcs-ready-dlami-with-enroot-pyxis.yaml`](../README.md#85-pre-baking-enrootpyxis-into-a-custom-ami)) when you want Enroot/Pyxis pre-baked or other customizations. See [OPERATIONS.md §2.5](./OPERATIONS.md#25-ami-selection-amiid--pin-in-production) |
| `SSHAccessCidr` | *(empty)* | When set to a CIDR, opens SSH/22 on the login node to that CIDR via a login-only security group (attached to the login node only, never compute). Empty (default) = SSH over SSM only. Set to your office/VPN range for direct `ssh`/`scp`/VS Code Remote (common for multi-user clusters) |
| `ManagedAccounting` | `disabled` | Enable Slurm managed accounting (requires Slurm 24.11+) |
| `AccountingPolicyEnforcement` | `none` | Slurm accounting policy enforcement (`none` or `associations,limits,safe`) |
| `GpuHealthCheck` | `none` | `prolog` wires the repo's [GPU health-check suite](../../../4.validation_and_observability/2.gpu-cluster-healthcheck) in as a cluster-wide Slurm **Prolog**: before each job it runs fast checks (nvidia-smi + EFA enumeration, ~8s) and **drains the node + requeues the job** if a GPU/EFA fault is detected. The prolog is GPU-aware (no-ops on CPU/login nodes), so it is safe on mixed clusters. The suite is installed to `/opt/aws/pcs/gpu-healthcheck` on the GPU compute node group at first boot (from `s3://<bucket>/<prefix>gpu-healthcheck/gpu-healthcheck.tar.gz`, so that object must be present in your templates bucket). PCS supports `Prolog` only at the cluster level, hence the GPU-aware self-scoping rather than a per-node-group setting. See [OPERATIONS.md §GPU health-check prolog](./OPERATIONS.md). |

## 3. On-Demand Compute Node Group (CPU)

> **Node-group cap.** AWS PCS limits a cluster to **10 compute node groups** and **10
> queues**, both **non-adjustable** internal quotas
> ([PCS quotas](https://docs.aws.amazon.com/pcs/latest/userguide/service-endpoints-quotas.html)).
> The login node group always uses one slot, so at most **9** GPU/CPU node groups can
> coexist. The default set is **login + 9 GPU queues** (see §4b–4e), which reaches the cap —
> so `DeployOnDemandCNG` (this CPU queue) defaults **`false`**. Enable it only if you also
> disable a GPU queue to keep the total ≤ 10; a template `Rule` fails the stack fast if the
> full default GPU set is combined with the CPU queue.

| Parameter | Default | Purpose |
|---|---|---|
| `DeployOnDemandCNG` | `false` | Deploy the CPU queue. Off by default because login + the 9 default GPU queues already reach the 10-node-group PCS cap — enable only if you disable a GPU queue |
| `OnDemandInstanceType` | `c6i.4xlarge` | CPU queue instance type |
| `OnDemandMinCount` | `0` | CPU queue minimum nodes (0 = dynamic scaling) |
| `OnDemandMaxCount` | `4` | CPU queue maximum nodes |
| `OnDemandCngName` | `cpu1` | CPU node-group name |
| `OnDemandQueueName` | `cpu1` | CPU Slurm queue name |
| `OnDemandEfaInterfaceCount` | `0` | EFA interfaces on the CPU CNG. **`0` (default) = no EFA** (standard ENA). `1` or `2` = enable EFA with that many interfaces (switches the LaunchTemplate to a `NetworkInterfaces` block with `InterfaceType=efa` + a cluster placement group). Set the count to the instance type's `MaximumEfaInterfaces`: `hpc8a.96xlarge`/`hpc7a.*`/`hpc6id.32xlarge`=2; `hpc6a.48xlarge`/`c7i.metal`=1. **EFA needs an EFA-capable type** — a non-EFA type (e.g. the default `c6i.4xlarge`) fails to launch with count > 0. No effect on the GPU CNG. See [README §8.6 CPU compute node group](../README.md#86-cpu-compute-node-group--advanced-settings) |
| `OnDemandPlacementGroupName` | *(empty)* | Existing cluster placement group name to launch nodes into. Empty + `OnDemandEfaInterfaceCount > 0` auto-creates a per-CNG cluster placement group; supplying a name reuses an existing one (e.g. shared across CPU + GPU CNGs for heterogeneous tightly-coupled jobs). Ignored when `OnDemandEfaInterfaceCount = 0` |

## 4. GPU Compute Node Group — P5/P6 (Optional)

See [GPU compute](../README.md#gpu-compute-p5p6) for instance/EFA/capacity guidance.

| Parameter | Default | Purpose |
|---|---|---|
| `DeployPseriesCNG` | `false` | Deploy a GPU (P5/P6) queue |
| `PseriesInstanceType` | `p5.48xlarge` | GPU instance type; selects the matching multi-NIC template **and** EFA interface count automatically |
| `PseriesMinCount` | `0` | GPU queue minimum nodes |
| `PseriesMaxCount` | `4` | GPU queue maximum nodes |
| `CapacityReservationId` | *(empty)* | Capacity **Block** reservation ID (sets `MarketType=capacity-block`). Leave empty for On-Demand / ODCR — **do not** put an ODCR ID here |
| `PseriesCngName` | `gpu-p5` | GPU node-group name |
| `PseriesQueueName` | `gpu-p5` | GPU Slurm queue name |

> **Spot P5/P6 — use the standalone template.** deploy-all's P-series path is On-Demand /
> ODCR / Capacity Block only (it does not pass a purchasing mode to the P5 child stack).
> For an interruptible **Spot** P5/P6 queue, deploy [`add-cng-p5.yaml`](../assets/add-cng-p5.yaml)
> as its own CNG stack against the running cluster (see [README Example 4](../README.md#example-4-multi-nic-gpu-spot-queue-p5en)).
> That template exposes three purchasing parameters (matching the `add-cng-g7e*.yaml` templates):
>
> | Parameter | Default | Purpose |
> |---|---|---|
> | `PurchaseOption` | `ONDEMAND` | `ONDEMAND` / `SPOT` / `CAPACITY_BLOCK`. Single source of truth for the node group's purchasing mode |
> | `SpotAllocationStrategy` | `price-capacity-optimized` | EC2 Spot allocation strategy when `PurchaseOption=SPOT` (`lowest-price` / `capacity-optimized` / `price-capacity-optimized`). Ignored otherwise |
> | `UsePlacementGroup` | `true` | Whether On-Demand nodes launch into an auto-created cluster placement group. Set `false` to relax placement and avoid `InsufficientInstanceCapacity` for scarce GPU types. No effect on Spot / Capacity Block (which never use one) |

## 4b–4e. GPU Queues — g7 / g7e / g6e (per-AZ, On-Demand)

Ten independent, separately-toggleable **On-Demand** GPU queues across three families,
each placed in the AZ where that family has the best capacity. Every queue is single-AZ
(the multi-NIC EFA launch templates pin the subnet per network interface, so a queue
cannot span AZs while keeping EFA); to run a family in more than one AZ, deploy its
per-AZ queues. `full` = 48xlarge, `half` = 24xlarge. `MinCount` stays `0`, so a queue costs
nothing until a job is submitted to its partition. These are independent of the P5/P6 queue
above.

> **PCS 10-node-group cap → 9 GPU queues on by default.** AWS PCS limits a cluster to **10
> compute node groups** and **10 queues** (both non-adjustable). The login node group always
> uses one slot, so all ten GPU queues **cannot** coexist with it. Therefore **nine** GPU
> queues default `true` and **`DeployG7Half` defaults `false`** (g7-full still covers the g7
> family in the primary AZ), giving login + 9 GPU = exactly 10 node groups. The CPU queue
> (`DeployOnDemandCNG`, §3) also defaults `false` for the same reason. To enable a
> tenth-plus queue (g7-half or CPU), **disable another queue first** to stay ≤ 10 — a
> template `Rule` fails the stack fast if the full default GPU set is combined with either.

**AZ placement.** The `-az2` / `-az3` queues launch into `AdditionalSubnetAZ2` /
`AdditionalSubnetAZ3` (section 1) and **require that subnet to be set** — if the AZ isn't
provided, the queue self-skips (its `Deploy*` default has no effect) instead of
failing the stack. Set `AdditionalSubnetAZ2` / `AdditionalSubnetAZ3` to the AZs where the
family has capacity (e.g. in `us-east-2`: g7 in most AZs, g7e in `az1`/`az2`, g6e in
`az2`/`az3`). See the [g7/g7e/g6e deploy guide](./G7E-DEPLOY.md).

Each family/size maps to a dedicated multi-NIC EFA template: g7 full = `add-cng-g7.yaml`
(2 NIC), g7 half = `add-cng-g7-24xl.yaml` (1 NIC), g7e full = `add-cng-g7e.yaml` (4 NIC),
g7e half = `add-cng-g7e-24xl.yaml` (2 NIC), g6e full = `add-cng-g6e.yaml` (4 NIC), g6e
half = `add-cng-g6e-24xl.yaml` (2 NIC).

| Parameter | Default | Purpose |
|---|---|---|
| `DeployG7Full` | `true` | Deploy the g7.48xlarge (8× RTX PRO 4500, 2-NIC EFA) queue in the primary AZ |
| `G7FullName` | `gpu-g7-full` | g7 full node-group + Slurm queue name |
| `G7FullMinCount` | `0` | g7 full minimum nodes (0 = dynamic scaling) |
| `G7FullMaxCount` | `2` | g7 full maximum nodes |
| `DeployG7Half` | `false` | Deploy the g7.24xlarge (4-GPU, 1-NIC EFA) queue in the primary AZ. **Off by default** to stay within the 10-node-group cap (login + 9 other GPU queues); enable only if you disable another queue |
| `G7HalfName` | `gpu-g7-half` | g7 half node-group + Slurm queue name |
| `G7HalfMinCount` | `0` | g7 half minimum nodes (0 = dynamic scaling) |
| `G7HalfMaxCount` | `2` | g7 half maximum nodes |
| `DeployG7eFull` | `true` | Deploy the g7e.48xlarge (8× RTX PRO 6000, 4-NIC EFA) queue in the primary AZ |
| `G7eFullName` | `gpu-g7e-full` | g7e full node-group + Slurm queue name |
| `G7eFullMinCount` | `0` | g7e full minimum nodes (0 = dynamic scaling) |
| `G7eFullMaxCount` | `2` | g7e full maximum nodes |
| `DeployG7eHalf` | `true` | Deploy the g7e.24xlarge (4-GPU, 2-NIC EFA) queue in the primary AZ |
| `G7eHalfName` | `gpu-g7e-half` | g7e half node-group + Slurm queue name |
| `G7eHalfMinCount` | `0` | g7e half minimum nodes (0 = dynamic scaling) |
| `G7eHalfMaxCount` | `2` | g7e half maximum nodes |
| `DeployG7eFullAz2` | `true` | Deploy a g7e.48xlarge queue in AZ2. **Requires `AdditionalSubnetAZ2`** (self-skips otherwise) |
| `G7eFullAz2Name` | `gpu-g7e-full-az2` | g7e full (AZ2) node-group + Slurm queue name |
| `G7eFullAz2MinCount` | `0` | g7e full (AZ2) minimum nodes (0 = dynamic scaling) |
| `G7eFullAz2MaxCount` | `2` | g7e full (AZ2) maximum nodes |
| `DeployG7eHalfAz2` | `true` | Deploy a g7e.24xlarge queue in AZ2. **Requires `AdditionalSubnetAZ2`** (self-skips otherwise) |
| `G7eHalfAz2Name` | `gpu-g7e-half-az2` | g7e half (AZ2) node-group + Slurm queue name |
| `G7eHalfAz2MinCount` | `0` | g7e half (AZ2) minimum nodes (0 = dynamic scaling) |
| `G7eHalfAz2MaxCount` | `2` | g7e half (AZ2) maximum nodes |
| `DeployG6eFullAz2` | `true` | Deploy a g6e.48xlarge (8× L40S, 4-NIC EFA) queue in AZ2. **Requires `AdditionalSubnetAZ2`** (self-skips otherwise) |
| `G6eFullAz2Name` | `gpu-g6e-full-az2` | g6e full (AZ2) node-group + Slurm queue name |
| `G6eFullAz2MinCount` | `0` | g6e full (AZ2) minimum nodes (0 = dynamic scaling) |
| `G6eFullAz2MaxCount` | `2` | g6e full (AZ2) maximum nodes |
| `DeployG6eHalfAz2` | `true` | Deploy a g6e.24xlarge (4-GPU, 2-NIC EFA) queue in AZ2. **Requires `AdditionalSubnetAZ2`** (self-skips otherwise) |
| `G6eHalfAz2Name` | `gpu-g6e-half-az2` | g6e half (AZ2) node-group + Slurm queue name |
| `G6eHalfAz2MinCount` | `0` | g6e half (AZ2) minimum nodes (0 = dynamic scaling) |
| `G6eHalfAz2MaxCount` | `2` | g6e half (AZ2) maximum nodes |
| `DeployG6eFullAz3` | `true` | Deploy a g6e.48xlarge queue in AZ3. **Requires `AdditionalSubnetAZ3`** (self-skips otherwise) |
| `G6eFullAz3Name` | `gpu-g6e-full-az3` | g6e full (AZ3) node-group + Slurm queue name |
| `G6eFullAz3MinCount` | `0` | g6e full (AZ3) minimum nodes (0 = dynamic scaling) |
| `G6eFullAz3MaxCount` | `2` | g6e full (AZ3) maximum nodes |
| `DeployG6eHalfAz3` | `true` | Deploy a g6e.24xlarge queue in AZ3. **Requires `AdditionalSubnetAZ3`** (self-skips otherwise) |
| `G6eHalfAz3Name` | `gpu-g6e-half-az3` | g6e half (AZ3) node-group + Slurm queue name |
| `G6eHalfAz3MinCount` | `0` | g6e half (AZ3) minimum nodes (0 = dynamic scaling) |
| `G6eHalfAz3MaxCount` | `2` | g6e half (AZ3) maximum nodes |
| `GpuUsePlacementGroup` | `true` | Shared across **all** g7/g7e/g6e queues. Launch On-Demand GPU nodes into a cluster placement group (`true`, lowest inter-node latency for tightly-coupled multi-node jobs). Set `false` to relax placement — a cluster placement group forces all nodes into one tight physical group, which can cause `InsufficientInstanceCapacity` for scarce GPU types even when the AZ has capacity; relaxing it improves On-Demand launch success (best for single-node jobs or when you hit ICE) |

> **Note — g7/g7e/g6e and Capacity Blocks.** These families are **not** eligible for
> Capacity Blocks for ML (those cover the P/Trn training families only), so the queues are
> On-Demand only — `CapacityReservationId` (section 4) does not apply to them.

## 5. Additional Cluster Configuration (Monitoring, Multi-User, Container Runtime)

| Parameter | Default | Purpose |
|---|---|---|
| `MonitoringStack` | `Prometheus-LoginNode` | Monitoring stack to deploy. `Prometheus-LoginNode` = self-hosted Prometheus + Grafana + DCGM Exporter on the login node. `none` = no monitoring. (Renamed from the old boolean `DeployMonitoring`; `<what>-<where>` enum, extensible to future `AMP-AMG`/`CloudWatch`) |
| `GrafanaAccessCidr` | *(empty)* | When set to a CIDR, opens HTTPS/443 (Grafana) on the login node to that CIDR via the login-only security group. Empty = SSM port-forward only. **443 also exposes the unauthenticated `/prometheus/`, `/pushgateway/`, `/slurmexporter/` proxy paths**, not just the password-gated Grafana. Use the tightest CIDR you can. (Renamed from `GrafanaPublicAccessCidr`) |
| `MonitoringRepo` | `aws-samples/aws-parallelcluster-monitoring` | GitHub `owner/repo` for the monitoring stack; override with a fork + a branch in `MonitoringVersion` to test unreleased changes |
| `MonitoringVersion` | `v2.9.1` | [aws-parallelcluster-monitoring](https://github.com/aws-samples/aws-parallelcluster-monitoring) git ref (release tag, branch, or `latest`). `v2.9.1` adds the `DCGM_EXPORTER_IMAGE` override (needed for B300 GPU metrics) and brings Grafana 13; `v2.6.4`+ carry the PCS `/opt` install + Docker-29.x DCGM fixes. Pin to a tag for stability. Migration notes: [OPERATIONS.md §3](./OPERATIONS.md#3-monitoring-monitoringversion) |
| `DcgmExporterImage` | DCGM 4.5.2 by digest | `dcgm-exporter` image used on GPU nodes. Defaults to a DCGM 4.5.2 build pinned by digest (`nvcr.io/nvidia/k8s/dcgm-exporter@sha256:a7ad6547...`) covering Hopper / B200 / B300. The digest pull bypasses the Docker-29.x OCI-index failure on newer NVCR tags. Override (any image reference, ideally also a digest) to pin to a different build — e.g. the monitoring stack's older default 4.2.0. No effect on CPU nodes. See [OPERATIONS.md §3.1](./OPERATIONS.md#31-dcgmexporterimage-the-default-and-when-to-change-it) |
| `DirectoryService` | `none` | Multi-user directory. `none` = single `ubuntu` user. `OpenLDAP-LoginNode` = slapd on the login node (DB on shared `/home/ldap-db`) + SSSD on all compute nodes. **Single login node only** — keep the login node group at 1 instance while enabled. See [USER-MANAGEMENT.md](./USER-MANAGEMENT.md) |
| `DirectoryDomainSuffix` | `dc=cluster,dc=internal` | LDAP domain suffix. Only used when `DirectoryService != none` |
| `PostInstallScriptUrl` | *(empty → auto)* | Script run on every node at first boot (PCS equivalent of ParallelCluster `OnNodeConfigured`). **Empty (default) auto-installs Enroot/Pyxis** from `s3://<S3BucketName>/<S3KeyPrefix>scripts/install-enroot-pyxis.sh` (fetched with the instance role, so it works with a **private** bucket — no public S3 needed). Accepts an `s3://` URL (instance-role fetch) or an `http(s)://` URL (curl, public only, e.g. GitHub raw). **Set to the literal `none` to skip** the hook entirely. (A single space does **not** skip — CloudFormation trims a whitespace-only value to empty, which selects the default installer.) Idempotent: the default installer is a no-op if Enroot/Pyxis is already pre-baked into `AmiId`, so leaving it empty is safe on a pre-baked AMI |
| `PostInstallScriptArgs` | *(empty)* | Arguments passed to the post-install script. Normally left empty — most users never touch the container-runtime parameters |

## 6. FSx Storage (`/fsx` and `/home`)

See [README §8.1 Storage](../README.md#81-storage-fsx-deployment-types--sizing) for Region
availability and the "deploy small, expand after" tip.

| Parameter | Default | Purpose |
|---|---|---|
| `Capacity` | `1200` | FSx for Lustre (`/fsx`) capacity (GiB; 1200 or increments of 2400). Can be increased after creation, so start small for a faster first deploy |
| `LustreDeploymentType` | `PERSISTENT_2` | FSx for Lustre (`/fsx`) deployment type (`PERSISTENT_2` / `PERSISTENT_1`) — Region-dependent |
| `PerUnitStorageThroughput` | `250` | FSx for Lustre (`/fsx`) throughput (MB/s/TiB); valid values depend on the deployment type |
| `Compression` | `LZ4` | FSx for Lustre (`/fsx`) data compression (`LZ4` / `NONE`) |
| `LustreVersion` | `2.15` | FSx for Lustre (`/fsx`) software version (`2.15` / `2.12`) |
| `FSxLustreEnableEfa` | `false` | Enable EFA on the FSx for Lustre filesystem. **The headline feature is GPUDirect Storage (GDS) for P5/P5e/P5en/P6-B200 GPU clients**, which DMAs file data straight into GPU memory (requires the NVIDIA `nvidia-fs` / cuFile stack on the client — tracked as a follow-up in [docs/ROADMAP.md](./ROADMAP.md)). EFA-capable CPU CNGs (`OnDemandEfaInterfaceCount > 0`) get the EFA *transport* path to storage as a secondary benefit, useful when a single client is pushing past ~10 GBps. **PERSISTENT_2 SSD only** — a CFN Rule on the prerequisites template fails the stack at create time when combined with PERSISTENT_1 (rather than silently ignoring the opt-in). **Requires a much larger `Capacity` than non-EFA**: at `PerUnitStorageThroughput=250` the minimum is **19200 GiB** (16× the 1200 GiB non-EFA default). The full minimum-capacity matrix per throughput tier is in the [FSx for Lustre User Guide](https://docs.aws.amazon.com/fsx/latest/LustreGuide/efa.html). The FSx side rejects undersized capacity at stack-create time with a clear error |
| `DataRepositoryS3Bucket` | `''` (none) | (Optional) Link an existing S3 bucket to the Lustre filesystem as a **Data Repository Association (DRA)**. When set, the bucket is mapped 1-1 with **`/fsx/s3`**: existing objects appear immediately (metadata batch-imported), new/changed/deleted S3 objects are lazily imported, and new/changed/deleted files under `/fsx/s3` are **auto-exported back** to the bucket (bidirectional). **Requirements:** the bucket must be in the **same Region** as the cluster (DRAs are single-Region — a cross-Region bucket cannot be linked) and `LustreDeploymentType` must be **`PERSISTENT_2`** (a CFN Rule fails the stack otherwise; subdir mapping + auto-export are PERSISTENT_2-only). FSx uses its own service-linked role to reach S3, so no instance-role or bucket-policy change is needed for a same-account bucket. Provide the **bare bucket name** (not an `s3://` URI). Empty (default) = no S3 link |
| `DataRepositoryS3Path` | `''` (whole bucket) | (Optional) Key prefix within `DataRepositoryS3Bucket` to link (e.g. `data/` links only `s3://<bucket>/data/` to `/fsx/s3`). Ignored when `DataRepositoryS3Bucket` is empty. Just the key prefix — no bucket name or `s3://` scheme |
| `HomeCapacity` | `512` | FSx for OpenZFS (`/home`) capacity (GiB). Can be increased after creation |
| `HomeThroughput` | `320` | FSx for OpenZFS (`/home`) throughput (MB/s) |
| `OpenZFSDeploymentType` | `SINGLE_AZ_HA_2` | FSx for OpenZFS (`/home`) deployment type (`SINGLE_AZ_HA_2` / `SINGLE_AZ_HA_1` / `SINGLE_AZ_2` / `SINGLE_AZ_1`) — Region-dependent |

## 7. Developer / Advanced

| Parameter | Default | Purpose |
|---|---|---|
| `S3BucketName` | `awsome-distributed-ai` | S3 bucket the nested templates are fetched from |
| `S3KeyPrefix` | `templates/` | S3 key prefix for the nested templates |
