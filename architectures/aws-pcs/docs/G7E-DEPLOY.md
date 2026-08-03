# Deploying a g7 / g7e / g6e GPU cluster

Quick guide for standing up an AWS PCS cluster with **g7**, **g7e**, and **g6e** GPU
nodes. Each family comes in two sizes, and each (family, size, AZ) combination is its own
independent On-Demand queue:

- **g7** — `g7.48xlarge` (8× NVIDIA RTX PRO 4500, 2 EFA NICs) via
  [`assets/add-cng-g7.yaml`](../assets/add-cng-g7.yaml) and `g7.24xlarge` (4 GPUs, 1 EFA NIC)
  via [`assets/add-cng-g7-24xl.yaml`](../assets/add-cng-g7-24xl.yaml)
- **g7e** — `g7e.48xlarge` (8× NVIDIA RTX PRO 6000, 4 EFA NICs) via
  [`assets/add-cng-g7e.yaml`](../assets/add-cng-g7e.yaml) and `g7e.24xlarge` (4 GPUs, 2 EFA NICs)
  via [`assets/add-cng-g7e-24xl.yaml`](../assets/add-cng-g7e-24xl.yaml)
- **g6e** — `g6e.48xlarge` (8× NVIDIA L40S, 4 EFA NICs) via
  [`assets/add-cng-g6e.yaml`](../assets/add-cng-g6e.yaml) and `g6e.24xlarge` (4 GPUs, 2 EFA NICs)
  via [`assets/add-cng-g6e-24xl.yaml`](../assets/add-cng-g6e-24xl.yaml)

## Per-AZ queues (why there are ten)

GPU capacity for these families is **AZ-specific**, and a multi-NIC EFA node group is
**single-AZ by construction**: the launch template pins the subnet inside each EFA network
interface, and AWS PCS requires the node group's subnet to match — so a single queue
cannot span AZs while keeping EFA. To run a family in more than one AZ, deploy a separate
queue per AZ. The template ships ten independent On-Demand queues:

| Queue (Slurm partition) | Instance | AZ slot | Default |
|---|---|---|---|
| `gpu-g7-full`      | g7.48xlarge  | primary (`PrimarySubnetAZ`) | on |
| `gpu-g7-half`      | g7.24xlarge  | primary | **off** |
| `gpu-g7e-full`     | g7e.48xlarge | primary | on |
| `gpu-g7e-half`     | g7e.24xlarge | primary | on |
| `gpu-g7e-full-az2` | g7e.48xlarge | AZ2 (`AdditionalSubnetAZ2`) | on |
| `gpu-g7e-half-az2` | g7e.24xlarge | AZ2 | on |
| `gpu-g6e-full-az2` | g6e.48xlarge | AZ2 | on |
| `gpu-g6e-half-az2` | g6e.24xlarge | AZ2 | on |
| `gpu-g6e-full-az3` | g6e.48xlarge | AZ3 (`AdditionalSubnetAZ3`) | on |
| `gpu-g6e-half-az3` | g6e.24xlarge | AZ3 | on |

> **PCS caps a cluster at 10 compute node groups and 10 queues (both non-adjustable).**
> The login node group always consumes one node-group slot, so all ten GPU queues can't
> coexist with it. **Nine** GPU queues default on and **`gpu-g7-half` defaults off**
> (g7-full still covers the g7 family in the primary AZ) — giving login + 9 GPU = exactly
> 10 node groups. The **CPU queue** (`DeployOnDemandCNG`) is also **off by default** for the
> same reason. To turn on `gpu-g7-half` or the CPU queue, **disable another GPU queue first**
> to stay ≤ 10 node groups; a template `Rule` fails the stack fast (with a clear message,
> not the opaque `ComputeNodeGroup` quota rollback) if the full default GPU set is combined
> with either.

Every queue is an independent toggle (`DeployG7Full`, `DeployG7eFullAz2`,
`DeployG6eHalfAz3`, …). `MinCount` stays `0`, so a queue costs nothing until a job is
submitted to its partition. The `-az2` / `-az3` queues additionally require their
`AdditionalSubnetAZ2` / `AdditionalSubnetAZ3` to be set — if the AZ isn't provided, that
queue **self-skips** instead of failing the stack.

It reuses the standard cluster: VPC + networking, FSx for Lustre (`/fsx` scratch),
FSx for OpenZFS (`/home`), and the Slurm scheduler. Enroot/Pyxis is installed at
first boot so containerized jobs work out of the box.

## 1. Prerequisites

- AWS CLI configured (`aws sts get-caller-identity`) with permissions to create
  CloudFormation, PCS, EC2, FSx, and IAM resources.
- **The instance types must actually launch in the AZs you pick.** Two independent gates
  have to clear, and both surface the same misleading Slurm symptom — `Node failure ...
  nodes are still not ready / Something is wrong with the boot of the nodes` — even though
  **no instance ever launched** (the EC2 request is rejected before boot):

  1. **Capacity (ICE).** EC2 must have spare capacity for that type in that AZ at scale-up
     time. A new/scarce GPU family often returns an *Insufficient Capacity Error* on
     On-Demand. First confirm the type is even offered per AZ:
     ```bash
     aws ec2 describe-instance-type-offerings --location-type availability-zone-id \
       --filters Name=instance-type,Values=g7e.48xlarge \
       --query 'InstanceTypeOfferings[].Location' --region <region> --output text
     ```
     Being offered ≠ available right now. If scale-up fails with no instance created:
     **set `GpuUsePlacementGroup=false`** (a cluster placement group concentrates all nodes
     in one tight physical group and is a common ICE trigger — relaxing it lets EC2 place
     nodes wherever there's capacity), move the family to a different AZ (its `-az2`/`-az3`
     queue), or retry later. **Capacity Blocks for ML do NOT apply to g7/g7e/g6e** — they
     cover the P/Trn training families only — so that is not a fallback for these types.
  2. **Quota.** g7/g7e/g6e are all **G/VT On-Demand** instances (48xlarge = 192 vCPUs,
     24xlarge = 96 vCPUs). Raise the On-Demand G/VT quota to `>= sum of (vCPUs × max nodes)`
     across all g7/g7e/g6e queues you enable:
     ```bash
     # On-Demand Running G/VT instances
     aws service-quotas get-service-quota --service-code ec2 \
       --quota-code L-DB2E81BA --region <region> --query 'Quota.Value' --output text
     ```
- **AZ capacity guidance (us-east-2 example).** Pick the AZ for each family where it has
  capacity. As observed in `us-east-2`: **g7** has capacity in most AZs; **g7e** in `az1`
  and `az2`; **g6e** in `az2` and `az3`. That maps cleanly onto the queue layout: g7 + g7e
  in the primary AZ (use `az1` or `az2`), g7e also in AZ2, and g6e in AZ2 + AZ3. Confirm
  live before deploying with `describe-instance-type-offerings` (above).
- An S3 bucket you own. Nested stacks are fetched by URL, so the templates must be
  staged in S3 (you can't deploy the parent with a local `--template-body` and have
  it find the child templates). The bucket can be private.

## 2. Stage the templates

Run from the repo's `architectures/aws-pcs` directory so `assets/` is the source.
This uploads every template (including the g7/g7e/g6e CNG templates) and the boot scripts:

```bash
cd architectures/aws-pcs
BUCKET=my-pcs-templates      # an S3 bucket you control
PREFIX=templates/            # keep the trailing slash

aws s3 sync assets/ "s3://${BUCKET}/${PREFIX}" \
  --exclude "*" --include "*.yaml" --include "*.sh"
```

> Always point the deploy at **your** bucket (step 3). The public
> `awsome-distributed-ai` bucket only holds published templates and can be a version
> behind, which causes nested-stack parameter-mismatch errors.

## 3. Deploy

Set the three AZs to where each family has capacity. The primary AZ carries the g7 and g7e
queues; AZ2 carries the g7e-az2 + g6e-az2 queues; AZ3 carries the g6e-az3 queues. The nine
default-on GPU queues (all except `gpu-g7-half`; see the node-group-cap note above) come up
from just the three AZ parameters.

> **Pick AZs by ID, not by name.** Availability Zone **names** (`us-east-2a`) are shuffled
> per account — `us-east-2a` in one account can be a different physical AZ than in another —
> so a hardcoded name won't reproduce the same capacity elsewhere. Availability Zone **IDs**
> (`use2-az1`) are stable across all accounts, and the capacity guidance above is stated in
> IDs. The template parameters need the *name* (they're typed
> `AWS::EC2::AvailabilityZone::Name`), so resolve ID → name at deploy time:
>
> ```bash
> az_name() { aws ec2 describe-availability-zones --region "$REGION" \
>   --filters "Name=zone-id,Values=$1" --query 'AvailabilityZones[0].ZoneName' --output text; }
> ```

```bash
REGION=us-east-2
# Target physical AZs by stable ID (portable across accounts):
AZ=$(az_name use2-az1)            # primary — g7 + g7e capacity
AZ2=$(az_name use2-az2)           # g7e + g6e capacity
AZ3=$(az_name use2-az3)           # g6e capacity
SSH_CIDR=203.0.113.4/32           # <-- your IP/CIDR for SSH to the login node

aws cloudformation create-stack \
  --stack-name pcs-gpu \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ml-cluster-deploy-all.yaml" \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ} \
    ParameterKey=AdditionalSubnetAZ2,ParameterValue=${AZ2} \
    ParameterKey=AdditionalSubnetAZ3,ParameterValue=${AZ3} \
    ParameterKey=S3BucketName,ParameterValue=${BUCKET} \
    ParameterKey=S3KeyPrefix,ParameterValue=${PREFIX} \
    ParameterKey=GpuUsePlacementGroup,ParameterValue=false \
    ParameterKey=LoginNodeInstanceType,ParameterValue=c7i.xlarge \
    ParameterKey=OnDemandInstanceType,ParameterValue=c7i.4xlarge \
    ParameterKey=SSHAccessCidr,ParameterValue=${SSH_CIDR} \
    ParameterKey=ManagedAccounting,ParameterValue=enabled \
    ParameterKey=DirectoryService,ParameterValue=OpenLDAP-LoginNode \
    ParameterKey=MonitoringStack,ParameterValue=none \
    ParameterKey=PostInstallScriptUrl,ParameterValue=none \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region ${REGION}
```

This brings up the nine default-on GPU Slurm partitions (all listed above except
`gpu-g7-half`, which defaults off to respect the 10-node-group cap — see the note above).
Each is an **independent toggle** — set any `DeployG7*` / `DeployG7e*` / `DeployG6e*` to
`false` to drop queues you don't want (e.g. keep only `DeployG6eFullAz3=true` for a single
g6e 48xlarge queue in AZ3). To enable `gpu-g7-half` or the CPU queue, disable another GPU
queue in the same command so the total stays ≤ 10 node groups (login + 9). If you omit
`AdditionalSubnetAZ2` / `AdditionalSubnetAZ3`, every queue pinned to that AZ self-skips and
you get a single-AZ cluster with just the g7 + g7e primary queues.

Useful parameter notes:

| Parameter | Value here | Notes |
|---|---|---|
| `AdditionalSubnetAZ2` / `AdditionalSubnetAZ3` | AZ names | Create the extra private subnets the `-az2` / `-az3` queues launch into. **Required** for those queues (they self-skip if empty) |
| `DeployG7Full` / `DeployG7Half` | `true` (default) | g7.48xlarge (2-NIC EFA) / g7.24xlarge (1-NIC EFA), primary AZ. Templates `add-cng-g7.yaml` / `add-cng-g7-24xl.yaml` |
| `DeployG7eFull` / `DeployG7eHalf` | `true` (default) | g7e.48xlarge (4-NIC EFA) / g7e.24xlarge (2-NIC EFA), primary AZ. Templates `add-cng-g7e.yaml` / `add-cng-g7e-24xl.yaml` |
| `DeployG7eFullAz2` / `DeployG7eHalfAz2` | `true` (default) | Same g7e templates, pinned to AZ2 |
| `DeployG6eFullAz2` / `DeployG6eHalfAz2` | `true` (default) | g6e.48xlarge (4-NIC EFA) / g6e.24xlarge (2-NIC EFA), AZ2. Templates `add-cng-g6e.yaml` / `add-cng-g6e-24xl.yaml` |
| `DeployG6eFullAz3` / `DeployG6eHalfAz3` | `true` (default) | Same g6e templates, pinned to AZ3 |
| `G7*/G7e*/G6e*MaxCount` | `2` | Per-queue max nodes; matching `…MinCount` defaults to 0 (scales from zero) |
| `GpuUsePlacementGroup` | `true` (default) | Applies to **all** g7/g7e/g6e queues. Set `false` to drop the cluster placement group — a cluster PG forces all nodes into one tight physical group and can cause `InsufficientInstanceCapacity` for scarce types; relaxing it improves launch success (best for single-node jobs; multi-node loses some latency locality) |
| `LoginNodeInstanceType` | `c7i.xlarge` | Login node size (default `m6i.4xlarge`) |
| `OnDemandInstanceType` | `c7i.4xlarge` | CPU queue node size (default `c6i.4xlarge`) |
| `SSHAccessCidr` | `203.0.113.4/32` | Opens SSH/22 on the login node to this CIDR. **Replace with your own IP/CIDR.** Empty (default) = SSM only |
| `ManagedAccounting` | `enabled` | Turns on Slurm accounting (per-user/job usage, `sacct`/`sreport`). Needs Slurm 24.11+ (the default 25.11 qualifies) |
| `DirectoryService` | `OpenLDAP-LoginNode` | Multi-user OpenLDAP on the login node + SSSD on compute nodes |
| `MonitoringStack` | `none` | No Prometheus/Grafana/DCGM. Drop this param (default `Prometheus-LoginNode`) to enable monitoring |
| `PostInstallScriptUrl` | `none` | Skips the first-boot Enroot/Pyxis install for faster boots during testing. **Leave this param off** (default empty) to auto-install Enroot/Pyxis — needed for containerized jobs (`srun --container-image=...`). Use the literal `none` to skip; a single space does **not** skip (CloudFormation trims it to empty → default installer) |
| `AmiId` | *(empty → DLAMI)* | Empty auto-resolves the latest PCS-Ready Ubuntu DLAMI. To run the GPU queues on **Rocky Linux 9** instead, build the Rocky AMI once and pass its `ami-xxx` here — see [Optional: Rocky Linux 9 AMI](#optional-run-on-rocky-linux-9) below |
| `DataRepositoryS3Bucket` | *(empty → no link)* | (Optional) Link an existing S3 bucket to the Lustre `/fsx` filesystem — its contents appear under `/fsx/s3` and changes sync back (bidirectional). See [Optional: link an S3 bucket to /fsx](#optional-link-an-s3-bucket-to-fsx) below |

> **Slurm accounting + multi-user.** `ManagedAccounting=enabled` pairs naturally with
> `DirectoryService=OpenLDAP-LoginNode` so usage is attributed per real user. To enforce
> limits (not just record usage), also add
> `ParameterKey=AccountingPolicyEnforcement,ParameterValue=associations,limits,safe`.
>
> **OpenLDAP runs on the single login node** — keep the login node group at one instance
> (this template already does) while it's enabled; the directory is a single point of
> failure if the login node is down. Add users with the helper on the login node — see
> [USER-MANAGEMENT.md](./USER-MANAGEMENT.md).

> **Note — g7/g7e/g6e and Capacity Blocks.** None of these families are eligible for
> Capacity Blocks for ML (those cover the P/Trn training families only), so the queues are
> On-Demand only — there is no Capacity-Block or Spot purchasing option for them here.

#### Changing queues on an already-deployed stack

You do **not** need to delete and recreate — `update-stack` adds or removes GPU queues
in place (each queue is its own nested stack; toggling one doesn't disturb the others or
the login/CPU nodes). First re-sync `assets/` to your bucket (step 2) so the updated
templates are published, then pass the `Deploy*` values you want and keep everything
else with `UsePreviousValue=true`:

```bash
aws cloudformation update-stack --stack-name pcs-gpu --region ${REGION} \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ml-cluster-deploy-all.yaml" \
  --parameters \
    ParameterKey=DeployG6eHalfAz3,ParameterValue=true \
    ParameterKey=G6eHalfAz3MaxCount,ParameterValue=2 \
    $(aws cloudformation describe-stacks --stack-name pcs-gpu --region ${REGION} \
        --query "Stacks[0].Parameters[].ParameterKey" --output text \
      | tr '\t' '\n' | grep -vx -e DeployG6eHalfAz3 -e G6eHalfAz3MaxCount \
      | sed 's/.*/ParameterKey=&,UsePreviousValue=true/') \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region ${REGION}
```

The shell snippet carries every other existing parameter forward unchanged; you only
spell out the ones you're changing. (Match `--stack-name` to your deployed stack.)

### Optional: run on Rocky Linux 9

By default every node group (login + g7/g7e/g6e) boots the AWS-published **PCS-Ready
Ubuntu 24.04 DLAMI**. To run the GPU queues on **Rocky Linux 9** instead, build a
PCS-Ready Rocky 9 GPU AMI once (a separate one-time stack, ~45–60 min) and pass its
`ami-xxx` as `AmiId`. The GPU queues, EFA wiring, and per-AZ layout are all identical —
only the OS image changes; the boot scripts auto-detect Rocky vs Ubuntu.

```bash
# One-time: build the Rocky 9 GPU AMI (see docs/ROCKY9-AMI.md for the full walkthrough).
# BASE_AMI = a kernel-updated official Rocky 9 cloud AMI; SubnetId/SecurityGroupIds are
# required in accounts with no default VPC.
aws cloudformation create-stack \
  --stack-name pcs-rocky9-ami \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ready-rocky9-gpu.yaml" \
  --parameters \
    ParameterKey=BaseAmiId,ParameterValue=${BASE_AMI} \
    ParameterKey=SlurmVersion,ParameterValue=25.11 \
    ParameterKey=SemanticVersion,ParameterValue=1.3.0 \
    ParameterKey=SubnetId,ParameterValue=${SUBNET_ID} \
    ParameterKey=SecurityGroupIds,ParameterValue=${SG_ID} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region ${REGION}

# Read the built AMI id, then pass it to the cluster deploy (step 3) as AmiId:
AMI_ID=$(aws cloudformation describe-stacks --stack-name pcs-rocky9-ami --region ${REGION} \
  --query 'Stacks[0].Outputs[?OutputKey==`Rocky9PCSAmiId`].OutputValue' --output text)

#   ...add to the create-stack --parameters in step 3:
#     ParameterKey=AmiId,ParameterValue=${AMI_ID} \
#     ParameterKey=SlurmVersion,ParameterValue=25.11 \    # match the AMI's Slurm (Pyxis ABI lock)
```

> The interactive user on Rocky is **`rocky`** (not `ubuntu`) — `sudo su - rocky` on the
> login node. Everything else in this guide (queues, SSM access, `sinfo`, jobs) is the same.
> Match `SlurmVersion` to what the AMI was built for. Full details and caveats:
> [ROCKY9-AMI.md](./ROCKY9-AMI.md).

### Optional: link an S3 bucket to /fsx

Set `DataRepositoryS3Bucket` to link an existing S3 bucket to the Lustre `/fsx`
filesystem via a **Data Repository Association (DRA)**. The bucket is mapped 1-1 with
**`/fsx/s3`**: existing objects appear immediately, new/changed/deleted S3 objects lazy-load
in, and files you create/change/delete under `/fsx/s3` **auto-export back** to the bucket
(bidirectional) — handy for staging datasets in and writing results out without a manual
`aws s3 cp` step.

Add to the step-3 `create-stack` parameters (or `update-stack` an existing cluster —
the DRA is a separate resource, so enabling it does **not** replace the filesystem):

```bash
    ParameterKey=DataRepositoryS3Bucket,ParameterValue=my-dataset-bucket \
    ParameterKey=DataRepositoryS3Path,ParameterValue= \   # optional key prefix; empty = whole bucket
```

Then on the login node the bucket contents are under `/fsx/s3`:

```bash
ls /fsx/s3/                                   # existing S3 objects (lazy-loaded)
echo hi > /fsx/s3/result.txt                  # auto-exports to s3://my-dataset-bucket/result.txt
```

> **Two hard requirements.** (1) The bucket must be in the **same Region** as the cluster —
> DRAs are single-Region, so a bucket in another Region cannot be linked. (2)
> `LustreDeploymentType` must be **`PERSISTENT_2`** (the default) — a template Rule fails the
> stack at create time otherwise. No bucket policy or extra IAM is needed for a same-account
> bucket (FSx uses its own service-linked role). Give the **bare bucket name**, not an
> `s3://` URI. Full parameter reference: [PARAMETERS.md](./PARAMETERS.md).

## 4. Monitor progress

```bash
aws cloudformation describe-stacks --stack-name pcs-gpu --region ${REGION} \
  --query 'Stacks[0].StackStatus' --output text
```

Typical create time is ~25–30 min (mostly VPC + FSx). When complete, the enabled GPU CNG
nested stacks are present (the P5/P6 stacks are not, unless you also enabled them).

## 5. Connect and run a job

Connect to the login node over SSM (no SSH key needed):

```bash
CLUSTER_ID=$(aws cloudformation describe-stacks --stack-name pcs-gpu --region ${REGION} \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterId`].OutputValue' --output text)
# The login node carries Name=PCS-login (PCS does NOT add a compute-node-group-name
# tag — it tags instances with aws:pcs:compute-node-group-id, an opaque pcs_xxxx ID).
# Scope by cluster id + the PCS-login Name tag, which is present with or without monitoring.
LOGIN_ID=$(aws ec2 describe-instances --region ${REGION} \
  --filters "Name=tag:pcs-cluster-id,Values=${CLUSTER_ID}" \
            "Name=tag:Name,Values=PCS-login" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
aws ssm start-session --target $LOGIN_ID --region ${REGION}
```

Then `sudo su - ubuntu` and check the queue / run a quick GPU job. (This deploy
disabled the Enroot/Pyxis install via `PostInstallScriptUrl=none`, so the
**container** workflow below is not yet available — these checks run directly on
the PCS-Ready DLAMI, which already has the NVIDIA driver, CUDA, and EFA stack.)

```bash
sinfo   # lists the nine default GPU partitions you deployed (gpu-g7-full, gpu-g7e-full-az2, gpu-g6e-half-az3, ...; gpu-g7-half only if you enabled it)

# Full (48xlarge): confirm all 8 GPUs are visible (scales a node up from 0; first launch ~3-5 min)
srun --partition=gpu-g7e-full --gpus-per-node=8 nvidia-smi -L

# Confirm the EFA fabric enumerates per family/size (full=4 for g7e/g6e, 2 for g7; half=2 for g7e/g6e, 1 for g7):
srun --partition=gpu-g7e-full fi_info -p efa | grep -c "provider: efa"   # expect 4
srun --partition=gpu-g7-full  fi_info -p efa | grep -c "provider: efa"   # expect 2
srun --partition=gpu-g7-half  fi_info -p efa | grep -c "provider: efa"   # expect 1

# A g6e node in AZ3, same idea (scales in the AZ3 subnet):
srun --partition=gpu-g6e-full-az3 --gpus-per-node=8 nvidia-smi -L
```

### Adding users

This deploy enabled `DirectoryService=OpenLDAP-LoginNode`, which stands up the
directory **structure** (the `People`/`Groups` OUs and a `clusterusers` group) but
**no login users** — only the LDAP admin DN. Create and manage users from the login
node (connect over SSM as above), following
**[USER-MANAGEMENT.md](./USER-MANAGEMENT.md)** — it covers fetching the LDAP admin
password, adding users with `ldap-add-user.sh`, authorizing SSH keys, and (since this
deploy set `ManagedAccounting=enabled`) registering users with Slurm accounting.

### Enabling containerized (NCCL) jobs later

The 2-node NCCL all-reduce — the quickest EFA-bandwidth check — runs through
Pyxis/Enroot, which this deploy skipped. To use it, redeploy (or
`update-stack`) **without** the `PostInstallScriptUrl` parameter so Enroot/Pyxis
auto-installs at boot, then on the login node:

```bash
# 2-node NCCL all-reduce over EFA (imports the container to shared /fsx)
TAG=cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9
enroot import -o /fsx/nccl-tests.sqsh "docker://public.ecr.aws#hpc-cloud/nccl-tests:${TAG}"
cd /fsx
wget https://raw.githubusercontent.com/awslabs/awsome-distributed-ai/main/micro-benchmarks/nccl-tests/slurm/nccl-tests-container.sbatch
sbatch --partition=gpu-g7e-full nccl-tests-container.sbatch
```

In the output, EFA is active when you see
`NET/OFI Selected provider is efa ... (found 4 nics)` (or the matching NIC count for the
partition), and a healthy run ends with `# Out of bounds values : 0 OK`.

> A stack update only re-runs first-boot scripts on **newly launched** nodes, so
> let a GPU queue scale a fresh node (or terminate existing ones) after
> re-enabling the install.

## 6. Clean up

```bash
aws cloudformation delete-stack --stack-name pcs-gpu --region ${REGION}
```

Nested stacks and the FSx filesystems are deleted automatically — **back up any
`/home` or `/fsx` data first**. If a compute-node-group stack hits `DELETE_FAILED`
(a PCS timing dependency), delete the PCS compute node groups first, then retry the
stack delete (see [DEPLOY-TESTING.md §7](./DEPLOY-TESTING.md#7-cleanup)).
