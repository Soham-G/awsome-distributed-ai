# Deploying a g7e GPU cluster

Quick guide for standing up an AWS PCS cluster with **g7e** GPU nodes. Two sizes are
available as separate queues, each in On-Demand and Spot flavors:
- **full** = `g7e.48xlarge` (8× NVIDIA RTX PRO 6000, 4 EFA NICs) via
  [`assets/add-cng-g7e.yaml`](../assets/add-cng-g7e.yaml)
- **half** = `g7e.24xlarge` (4 GPUs, 2 EFA NICs) via
  [`assets/add-cng-g7e-24xl.yaml`](../assets/add-cng-g7e-24xl.yaml)

Each queue is an independent toggle (`DeployG7eFull` / `DeployG7eFullSpot` /
`DeployG7eHalf` / `DeployG7eHalfSpot`) — deploy any combination.

It reuses the standard cluster: VPC + networking, FSx for Lustre (`/fsx` scratch),
FSx for OpenZFS (`/home`), and the Slurm scheduler. Enroot/Pyxis is installed at
first boot so containerized jobs work out of the box.

## 1. Prerequisites

- AWS CLI configured (`aws sts get-caller-identity`) with permissions to create
  CloudFormation, PCS, EC2, FSx, and IAM resources.
- **g7e.48xlarge must actually launch in your AZ.** Two independent gates have to
  clear, and both surface the same misleading Slurm symptom — `Node failure ... nodes
  are still not ready / Something is wrong with the boot of the nodes` — even though
  **no instance ever launched** (the EC2 request is rejected before boot):

  1. **Capacity (ICE).** EC2 must have spare g7e capacity in your AZ at scale-up time.
     A new/scarce GPU family often returns an *Insufficient Capacity Error* on
     On-Demand. First confirm the type is even offered in your AZ:
     ```bash
     aws ec2 describe-instance-type-offerings --location-type availability-zone \
       --filters Name=instance-type,Values=g7e.48xlarge \
       --query 'InstanceTypeOfferings[].Location' --region <region> --output text
     ```
     Being offered ≠ available right now. If scale-up fails with no instance created:
     **set `G7eUsePlacementGroup=false`** (a cluster placement group concentrates all
     nodes in one tight physical group and is a common ICE trigger — relaxing it lets
     EC2 place nodes wherever there's capacity), try the other AZ (`PrimarySubnetAZ`),
     or retry later. **Capacity Blocks for ML do NOT apply to g7e** — Capacity Blocks
     cover the P/Trn training families only (verified: `describe-capacity-block-offerings`
     rejects `g7e.48xlarge` with "not supported for Capacity Blocks"), so that is not a
     fallback for this instance type.
  2. **Quota.** g7e.48xlarge is **192 vCPUs**, g7e.24xlarge is **96 vCPUs**, and
     On-Demand vs Spot use *separate* quotas — raise whichever purchasing mode you're
     using to `>= sum of (vCPUs × max nodes)` across the queues of that mode:
     ```bash
     # On-Demand G/VT  (DeployG7eFull + DeployG7eHalf nodes)
     aws service-quotas get-service-quota --service-code ec2 \
       --quota-code L-DB2E81BA --region <region> --query 'Quota.Value' --output text
     # Spot G/VT  (DeployG7eFullSpot + DeployG7eHalfSpot nodes) — Client.MaxSpotInstanceCountExceeded if too low
     aws service-quotas get-service-quota --service-code ec2 \
       --quota-code L-3819A6DF --region <region> --query 'Quota.Value' --output text
     ```
- An S3 bucket you own. Nested stacks are fetched by URL, so the templates must be
  staged in S3 (you can't deploy the parent with a local `--template-body` and have
  it find the child templates). The bucket can be private.

## 2. Stage the templates

Run from the repo's `architectures/aws-pcs` directory so `assets/` is the source.
This uploads every template (including `add-cng-g7e.yaml`) and the boot scripts:

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

```bash
REGION=us-east-2
AZ=us-east-2a
SSH_CIDR=85.8.167.188/32          # <-- your IP/CIDR for SSH to the login node

aws cloudformation create-stack \
  --stack-name pcs-g7e \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ml-cluster-deploy-all.yaml" \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${AZ} \
    ParameterKey=S3BucketName,ParameterValue=${BUCKET} \
    ParameterKey=S3KeyPrefix,ParameterValue=${PREFIX} \
    ParameterKey=DeployG7eFull,ParameterValue=true \
    ParameterKey=G7eFullMaxCount,ParameterValue=2 \
    ParameterKey=DeployG7eFullSpot,ParameterValue=true \
    ParameterKey=G7eFullSpotMaxCount,ParameterValue=2 \
    ParameterKey=DeployG7eHalf,ParameterValue=true \
    ParameterKey=G7eHalfMaxCount,ParameterValue=2 \
    ParameterKey=DeployG7eHalfSpot,ParameterValue=true \
    ParameterKey=G7eHalfSpotMaxCount,ParameterValue=2 \
    ParameterKey=G7eUsePlacementGroup,ParameterValue=false \
    ParameterKey=LoginNodeInstanceType,ParameterValue=c7i.xlarge \
    ParameterKey=OnDemandInstanceType,ParameterValue=c7i.4xlarge \
    ParameterKey=SSHAccessCidr,ParameterValue=${SSH_CIDR} \
    ParameterKey=ManagedAccounting,ParameterValue=enabled \
    ParameterKey=DirectoryService,ParameterValue=OpenLDAP-LoginNode \
    ParameterKey=MonitoringStack,ParameterValue=none \
    ParameterKey=PostInstallScriptUrl,ParameterValue=" " \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region ${REGION}
```

This brings up four g7e Slurm partitions: `gpu-g7e-full` (48xlarge On-Demand),
`gpu-g7e-full-spot` (48xlarge Spot), `gpu-g7e-half` (24xlarge On-Demand), and
`gpu-g7e-half-spot` (24xlarge Spot). Each is an **independent toggle** — drop any
`DeployG7e*` line you don't want (e.g. keep only `DeployG7eHalf=true` for a single
24xlarge On-Demand queue).

Useful parameter notes:

| Parameter | Value here | Notes |
|---|---|---|
| `DeployG7eFull` / `…FullSpot` | `true` | g7e.48xlarge (8 GPUs, 4-NIC EFA) On-Demand / Spot queues. Template `add-cng-g7e.yaml` |
| `DeployG7eHalf` / `…HalfSpot` | `true` | g7e.24xlarge (4 GPUs, 2-NIC EFA) On-Demand / Spot queues. Template `add-cng-g7e-24xl.yaml` |
| `G7e*MaxCount` | `2` | Per-queue max nodes; matching `…MinCount` defaults to 0 (scales from zero) |
| `G7eUsePlacementGroup` | `true` (default) | Applies to **both On-Demand** g7e queues. Set `false` to drop the cluster placement group — a cluster PG forces all nodes into one tight physical group and can cause `InsufficientInstanceCapacity` for scarce types like g7e; relaxing it improves launch success (best for single-node jobs; multi-node loses some latency locality). No effect on the Spot queues |
| `G7eSpotAllocationStrategy` | *(default `price-capacity-optimized`)* | Applies to **both Spot** g7e queues. `price-capacity-optimized` (recommended) / `capacity-optimized` / `lowest-price` |
| `LoginNodeInstanceType` | `c7i.xlarge` | Login node size (default `m6i.4xlarge`) |
| `OnDemandInstanceType` | `c7i.4xlarge` | CPU queue node size (default `c6i.4xlarge`) |
| `SSHAccessCidr` | `203.0.113.4/32` | Opens SSH/22 on the login node to this CIDR. **Replace with your own IP/CIDR.** Empty (default) = SSM only |
| `ManagedAccounting` | `enabled` | Turns on Slurm accounting (per-user/job usage, `sacct`/`sreport`). Needs Slurm 24.11+ (the default 25.11 qualifies) |
| `DirectoryService` | `OpenLDAP-LoginNode` | Multi-user OpenLDAP on the login node + SSSD on compute nodes |
| `MonitoringStack` | `none` | No Prometheus/Grafana/DCGM. Drop this param (default `Prometheus-LoginNode`) to enable monitoring |
| `PostInstallScriptUrl` | `" "` (a single space) | Skips the first-boot Enroot/Pyxis install for faster boots during testing. **Leave this param off** (default empty) to auto-install Enroot/Pyxis — needed for containerized jobs (`srun --container-image=...`) |

> **Slurm accounting + multi-user.** `ManagedAccounting=enabled` pairs naturally with
> `DirectoryService=OpenLDAP-LoginNode` so usage is attributed per real user. To enforce
> limits (not just record usage), also add
> `ParameterKey=AccountingPolicyEnforcement,ParameterValue=associations,limits,safe`.
>
> **OpenLDAP runs on the single login node** — keep the login node group at one instance
> (this template already does) while it's enabled; the directory is a single point of
> failure if the login node is down. Add users with the helper on the login node — see
> [USER-MANAGEMENT.md](./USER-MANAGEMENT.md).

> **Spot nodes can be reclaimed at any time** (EC2 gives a 2-minute warning). For the
> `*-spot` queues, checkpoint long jobs and submit with `--requeue` so Slurm requeues
> an interrupted job instead of failing it. Spot draws on a separate capacity pool from
> On-Demand (so it sidesteps the On-Demand G/VT quota — but Spot has its **own** G/VT
> quota), and g7e Spot capacity can also be scarce; if a pool is empty that queue simply
> won't scale (no node launches).

> **Note — g7e and Capacity Blocks.** g7e is **not** eligible for Capacity Blocks for ML
> (those cover the P/Trn training families only), so there is no Capacity-Block option
> for these queues — On-Demand and Spot are the only purchasing modes.

#### Changing queues on an already-deployed stack

You do **not** need to delete and recreate — `update-stack` adds or removes g7e queues
in place (each queue is its own nested stack; toggling one doesn't disturb the others or
the login/CPU nodes). First re-sync `assets/` to your bucket (step 2) so the updated
templates are published, then pass the `DeployG7e*` values you want and keep everything
else with `UsePreviousValue=true`:

```bash
aws cloudformation update-stack --stack-name pcs-g7e --region ${REGION} \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ml-cluster-deploy-all.yaml" \
  --parameters \
    ParameterKey=DeployG7eHalf,ParameterValue=true \
    ParameterKey=G7eHalfMaxCount,ParameterValue=2 \
    $(aws cloudformation describe-stacks --stack-name pcs-g7e --region ${REGION} \
        --query "Stacks[0].Parameters[].ParameterKey" --output text \
      | tr '\t' '\n' | grep -vx -e DeployG7eHalf -e G7eHalfMaxCount \
      | sed 's/.*/ParameterKey=&,UsePreviousValue=true/') \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region ${REGION}
```

The shell snippet carries every other existing parameter forward unchanged; you only
spell out the ones you're changing. (Match `--stack-name` to your deployed stack.)

## 4. Monitor progress

```bash
aws cloudformation describe-stacks --stack-name pcs-g7e --region ${REGION} \
  --query 'Stacks[0].StackStatus' --output text
```

Typical create time is ~25–30 min (mostly VPC + FSx). When complete, the nested
`G7eCNGStack` is present (the P5/P6 stacks are not).

## 5. Connect and run a job

Connect to the login node over SSM (no SSH key needed):

```bash
CLUSTER_ID=$(aws cloudformation describe-stacks --stack-name pcs-g7e --region ${REGION} \
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
disabled the Enroot/Pyxis install via `PostInstallScriptUrl=" "`, so the
**container** workflow below is not yet available — these checks run directly on
the PCS-Ready DLAMI, which already has the NVIDIA driver, CUDA, and EFA stack.)

```bash
sinfo   # lists the g7e partitions you deployed: gpu-g7e-full / -full-spot / -half / -half-spot

# Full (48xlarge): confirm all 8 GPUs are visible (scales a node up from 0; first launch ~3-5 min)
srun --partition=gpu-g7e-full --gpus-per-node=8 nvidia-smi -L

# Full has 4 EFA cards; half (24xlarge) has 2. Confirm the fabric enumerates:
srun --partition=gpu-g7e-full fi_info -p efa | grep -c "provider: efa"   # expect 4
srun --partition=gpu-g7e-half fi_info -p efa | grep -c "provider: efa"   # expect 2

# A Spot node, same idea (may not scale if the g7e Spot pool is empty):
srun --partition=gpu-g7e-full-spot --gpus-per-node=8 nvidia-smi -L
```

### Adding the first user (SSH login)

This deploy enabled `DirectoryService=OpenLDAP-LoginNode`, which stands up the
directory **structure** (the `People`/`Groups` OUs and a `clusterusers` group) but
**no login users** — only the LDAP admin DN. Add users from the login node with the
`ldap-add-user.sh` helper (installed at `/usr/local/bin/`). Run these on the login
node (over SSM, as in the connect step above):

```bash
# The PCS cluster ID — read from this login node's own instance tag
CLUSTER_ID=$(curl -s http://169.254.169.254/latest/meta-data/tags/instance/pcs-cluster-id)

# Fetch the auto-generated LDAP admin password (needed by the helper)
export LDAP_ADMIN_PASSWORD=$(aws ssm get-parameter \
  --name "/pcs/${CLUSTER_ID}/ldap/admin-password" \
  --with-decryption --query 'Parameter.Value' --output text)

# Create a POSIX user. Args: <username> <uid> [gid=3000] [ssh-pub-key]
# uid is REQUIRED and must be cluster-unique (>= 1001); gid 3000 = clusterusers.
sudo -E /usr/local/bin/ldap-add-user.sh alice 10001 3000

# Confirm the user now resolves cluster-wide (SSSD)
getent passwd alice
```

The helper prints a random initial password for the account. (`CLUSTER_ID` is also
the stack's `ClusterId` output if you'd rather pass it explicitly.)

**Authorize an SSH key.** The home dir is auto-created on first login, but to log
in by key you seed `~/.ssh/authorized_keys` on the shared `/home` (OpenZFS) from
the login node — SSSD is not configured to serve SSH keys from LDAP, so the
helper's optional 4th key arg is stored but not used by `sshd`:

```bash
sudo install -d -m 700 -o alice -g clusterusers /home/alice/.ssh
echo "ssh-ed25519 AAAA... you@laptop" | sudo tee /home/alice/.ssh/authorized_keys
sudo chown alice:clusterusers /home/alice/.ssh/authorized_keys
sudo chmod 600 /home/alice/.ssh/authorized_keys
```

Then SSH in directly from a host inside your `SSHAccessCidr` (the login node has a
public IP and SSH/22 open to that CIDR):

```bash
LOGIN_IP=$(aws ec2 describe-instances --region ${REGION} \
  --filters "Name=tag:pcs-cluster-id,Values=${CLUSTER_ID}" \
            "Name=tag:Name,Values=PCS-login" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
ssh alice@${LOGIN_IP}
```

Because Slurm accounting is enabled (`ManagedAccounting=enabled`), also register the
user with the accounting DB so their jobs are tracked:

```bash
sudo sacctmgr -i add user alice account=default
```

See [USER-MANAGEMENT.md](./USER-MANAGEMENT.md) for groups, removing users, and
Slurm accounting details.

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
`NET/OFI Selected provider is efa ... (found 4 nics)`, and a healthy run ends with
`# Out of bounds values : 0 OK`.

> A stack update only re-runs first-boot scripts on **newly launched** nodes, so
> let the GPU queue scale a fresh node (or terminate existing ones) after
> re-enabling the install.

## 6. Clean up

```bash
aws cloudformation delete-stack --stack-name pcs-g7e --region ${REGION}
```

Nested stacks and the FSx filesystems are deleted automatically — **back up any
`/home` or `/fsx` data first**. If a compute-node-group stack hits `DELETE_FAILED`
(a PCS timing dependency), delete the PCS compute node groups first, then retry the
stack delete (see [DEPLOY-TESTING.md §7](./DEPLOY-TESTING.md#7-cleanup)).
