# Building and using a PCS-Ready Rocky Linux 9 GPU AMI

By default every node group in this reference architecture boots the AWS-published
**PCS-Ready Ubuntu 24.04 DLAMI** (PCS agent, Slurm, NVIDIA/CUDA, and EFA pre-baked). To run
the G-series GPU queues on **Rocky Linux 9** instead, build a PCS-Ready Rocky 9 GPU AMI with
[`pcs-ready-rocky9-gpu.yaml`](../assets/pcs-ready-rocky9-gpu.yaml) and pass its `ami-xxx` as
`AmiId`. Rocky Linux 9 is an
[officially supported AWS PCS operating system](https://docs.aws.amazon.com/pcs/latest/userguide/operating-systems.html).

Unlike the Ubuntu DLAMI path (see [CUSTOM-AMI.md](./CUSTOM-AMI.md), which only *layers*
Enroot/Pyxis onto an already-PCS-Ready base), there is **no AWS-published PCS-Ready Rocky
base**. This template builds the whole stack from a stock Rocky 9 cloud image using AWS's own
installers, in this order (EC2 Image Builder reboots between kmod layers):

1. **Kernel update + toolchain** — `dnf -y update`, EPEL, Development Tools, matching
   `kernel-devel`, the SSM agent, and the **AWS CLI v2** (`awscli2`); reboot. The AWS CLI is
   load-bearing for the boot path, not just build tooling — stock Rocky 9 has no `aws` (unlike
   the Ubuntu DLAMI), and the first-boot scripts use it for `aws s3 cp` (post-install /
   Enroot-Pyxis, GPU health-check) and `aws ssm` (monitoring Grafana secret). Without it those
   steps fail `aws: command not found`. Baked in from **v1.3.0** onward; the CNG UserData also
   installs it if missing so older AMIs self-heal at first boot.
2. **NVIDIA driver + CUDA + container toolkit** — from NVIDIA's rhel9 CUDA dnf repo.
3. **EFA** — the `aws-efa-installer` (GPG-verified).
4. **AWS PCS agent** — the AWS agent installer (GPG-verified). *Required* for the node to
   register with PCS.
5. **Slurm** — the AWS Slurm installer (GPG-verified), version-locked to `SlurmVersion`.
   Installs to `/opt/aws/pcs/scheduler/slurm-<ver>` — the exact layout the rest of the repo
   keys on.
6. **Enroot + Pyxis** — Enroot `.rpm`, `libnvidia-container` yum repo, Pyxis compiled against
   this Slurm version.
7. **DCGM** — for the monitoring stack.
8. **FSx Lustre client** — el9 kmod for `/fsx`.
9. **SELinux** — set the workload booleans and bake the chosen mode.

## Prerequisites

- **A Rocky 9 base AMI** (`BaseAmiId`). Rocky publishes official cloud images per Region under
  AWS account **`792107900819`** — resolve the latest for your Region with the query in Step 1
  (no Marketplace subscription needed). **Kernel currency matters:** PCS custom-AMI builds can
  fail on a stale base kernel (the NVIDIA/EFA/Lustre kmods must match the running kernel). The
  published base is often a point release behind; the build runs `dnf -y update` as a safety net
  (this is what produced the working AMIs), but for maximum reliability launch the base,
  `sudo dnf -y update`, reboot, re-image, and pass *that* AMI as `BaseAmiId` instead.
- **A build subnet + security group** (`SubnetId` / `SecurityGroupIds`). Leave both empty to use
  the account's **default VPC**. Accounts with **no default VPC** must pass a subnet — Image
  Builder's `LaunchBuildInstance` step fails to place the build instance otherwise. Use a subnet
  with outbound internet on 443 (public with an IGW route, or private + NAT) and a security group
  that allows egress to `0.0.0.0/0:443`. Step 1 shows how to find one.
- Permissions to create EC2 Image Builder, IAM, and EC2 resources.
- Build-instance egress on 443 to: the regional `aws-pcs-repo` bucket, `efa-installer.amazonaws.com`,
  NVIDIA's `developer.download.nvidia.com` + `nvidia.github.io`, the FSx Lustre client repo,
  GitHub, and the Rocky/EPEL mirrors.

## Step 1: Build the AMI (~45–60 min one-time, separate stack)

First stage the template in an S3 bucket you control (the same bucket you'll deploy the
cluster from), then resolve the base AMI and the build network, then launch the build.

```bash
REGION=us-east-2
BUCKET=my-pcs-templates       # an S3 bucket you control (can be private)
PREFIX=templates/             # key prefix (keep the trailing slash)

# Stage the builder template (and the rest of assets/) to your bucket:
#   run from the repo's architectures/aws-pcs directory
aws s3 sync assets/ "s3://${BUCKET}/${PREFIX}" --exclude "*" --include "*.yaml" --include "*.sh"
```

**Resolve the latest official Rocky 9 base AMI for this Region** (owner `792107900819` is
Rocky's official AWS account; AMI IDs are Region-specific, so this auto-picks the right one):

```bash
BASE_AMI=$(aws ec2 describe-images --region "$REGION" \
  --owners 792107900819 \
  --filters "Name=name,Values=Rocky-9-EC2-Base-*x86_64*" \
            "Name=state,Values=available" "Name=architecture,Values=x86_64" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
echo "BASE_AMI=$BASE_AMI"
```

**Pick the build subnet + security group.** In an account with a default VPC you can skip this
and leave both parameters empty. With **no default VPC**, resolve a subnet that has outbound
internet on 443 (public + IGW route, or private + NAT) and a security group in the same VPC that
allows egress on 443:

```bash
# A public subnet (auto-assigns a public IP → simplest egress path):
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=map-public-ip-on-launch,Values=true" \
  --query 'Subnets[0].SubnetId' --output text)

# The default security group of that subnet's VPC (its default allow-all egress is sufficient):
VPC_ID=$(aws ec2 describe-subnets --region "$REGION" --subnet-ids "$SUBNET_ID" \
  --query 'Subnets[0].VpcId' --output text)
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" \
  --query 'SecurityGroups[0].GroupId' --output text)
echo "SUBNET_ID=$SUBNET_ID  SG_ID=$SG_ID  (VPC $VPC_ID)"
```

> Sanity-check the subnet actually reaches the internet before a ~45-min build: a **public**
> subnet needs `map-public-ip-on-launch=true` **and** a `0.0.0.0/0` route to an IGW; a **private**
> subnet needs a `0.0.0.0/0` route to a NAT gateway. The command above picks a public one.

**Launch the build** (omit the `SubnetId` / `SecurityGroupIds` lines to use the default VPC):

```bash
aws cloudformation create-stack \
  --stack-name pcs-rocky9 \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ready-rocky9-gpu.yaml" \
  --parameters \
    ParameterKey=BaseAmiId,ParameterValue=${BASE_AMI} \
    ParameterKey=SlurmVersion,ParameterValue=25.11 \
    ParameterKey=SemanticVersion,ParameterValue=1.3.0 \
    ParameterKey=SubnetId,ParameterValue=${SUBNET_ID} \
    ParameterKey=SecurityGroupIds,ParameterValue=${SG_ID} \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region ${REGION}
```

Bump `SemanticVersion` on **every** rebuild — reusing a version makes EC2 Image Builder reuse
the cached component build, so template edits silently don't take effect.

Like the DLAMI template, the AMI is **single-Slurm-version by design** (Pyxis' SPANK ABI is
locked to its compile-time Slurm) — pass the same `SlurmVersion` you'll use on the cluster.
Pin `PcsAgentVersion` / `SlurmInstallerVersion` / `NvidiaDriverBranch` / `EfaInstallerVersion`
explicitly for reproducible builds; verify the agent/Slurm checksums against the
[AWS PCS software installers](https://docs.aws.amazon.com/pcs/latest/userguide/working-with_ami_installers.html)
page.

> **SELinux.** Rocky 9 ships SELinux **enforcing**. The template's `SELinuxMode` parameter
> defaults to **`permissive`** so the AMI works reliably out of the gate (would-be denials
> are logged as AVCs, not enforced) — the PCS workload spans NFS `/home`, Lustre `/fsx`,
> Enroot/containers, and slurmd, and getting every context right under enforcing takes
> iteration. To harden later, launch a node, exercise the workload, review
> `ausearch -m avc`, add the needed policy, then rebuild with `SELinuxMode=enforcing`.

## Step 2: Read the resulting AMI ID

```bash
AMI_ID=$(aws cloudformation describe-stacks \
  --stack-name pcs-rocky9 --region ${REGION} \
  --query 'Stacks[0].Outputs[?OutputKey==`Rocky9PCSAmiId`].OutputValue' \
  --output text)
echo "$AMI_ID"
```

## Step 3: Pass it to the cluster as `AmiId`

Enroot/Pyxis is baked in, so you can skip the boot-time install by passing
`PostInstallScriptUrl=none` (as with the DLAMI path). The boot scripts
(`install-enroot-pyxis.sh`, `setup-directory.sh`) auto-detect Rocky vs Ubuntu, so leaving
`PostInstallScriptUrl` at its default is also safe (the default installer is an idempotent
no-op on a pre-baked AMI). Note: a single space does **not** skip — CloudFormation trims a
whitespace-only value to empty, which runs the default installer.

Reusing the `REGION` / `BUCKET` / `PREFIX` and `AMI_ID` from Steps 1–2:

```bash
aws cloudformation create-stack \
  --stack-name pcs-gpu \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ml-cluster-deploy-all.yaml" \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-east-2a \
    ParameterKey=S3BucketName,ParameterValue=${BUCKET} \
    ParameterKey=S3KeyPrefix,ParameterValue=${PREFIX} \
    ParameterKey=AmiId,ParameterValue=$AMI_ID \
    ParameterKey=SlurmVersion,ParameterValue=25.11 \
    ParameterKey=PostInstallScriptUrl,ParameterValue=none \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region ${REGION}
```

Match `SlurmVersion` to what the AMI was built for (the Pyxis ABI lock). Pass `S3BucketName`
(and `S3KeyPrefix` if you changed it) so the nested stacks and boot scripts are fetched from
**your** bucket, not the public one.

## Deploy a full Rocky 9 cluster (verified end-to-end)

The AMI is a drop-in for the standard cluster template — pass it as `AmiId`. This was
validated on a live deploy (us-east-2, Rocky 9.8): all node groups came up **ACTIVE** and
the login node registered with SSM. The boot scripts auto-detect Rocky vs Ubuntu, so the
same `pcs-ml-cluster-deploy-all.yaml` works unchanged.

Reusing `REGION` / `BUCKET` / `PREFIX` / `AMI_ID` from Steps 1–2 (or set them here):

```bash
REGION=us-east-2
BUCKET=my-pcs-templates       # the bucket you staged the templates in (Step 1)
PREFIX=templates/
AMI_ID=ami-xxxxxxxxxxxxxxxxx  # the Rocky9PCSAmiId output from Step 2

# AZs by stable ID (portable across accounts) -> names the template needs:
az() { aws ec2 describe-availability-zones --region "$REGION" \
  --filters "Name=zone-id,Values=$1" --query 'AvailabilityZones[0].ZoneName' --output text; }

aws cloudformation create-stack \
  --stack-name pcs-rocky9-cluster \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}pcs-ml-cluster-deploy-all.yaml" \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=$(az use2-az1) \
    ParameterKey=AdditionalSubnetAZ2,ParameterValue=$(az use2-az2) \
    ParameterKey=AdditionalSubnetAZ3,ParameterValue=$(az use2-az3) \
    ParameterKey=S3BucketName,ParameterValue=${BUCKET} \
    ParameterKey=S3KeyPrefix,ParameterValue=${PREFIX} \
    ParameterKey=AmiId,ParameterValue=${AMI_ID} \
    ParameterKey=SlurmVersion,ParameterValue=25.11 \
    ParameterKey=PostInstallScriptUrl,ParameterValue=none \
    ParameterKey=GpuUsePlacementGroup,ParameterValue=false \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region ${REGION}
```

> To also link an S3 bucket to `/fsx` (same-Region bucket, contents appear under `/fsx/s3`,
> bidirectional), add `ParameterKey=DataRepositoryS3Bucket,ParameterValue=<bucket>` — see
> [PARAMETERS.md](./PARAMETERS.md) and the g7/g7e/g6e guide's
> [S3-link section](./G7E-DEPLOY.md#optional-link-an-s3-bucket-to-fsx).

**SSM on Rocky is wired up automatically:** the AMI installs the SSM agent, but EC2 Image
Builder strips it from the output AMI during its cleanup (Image Builder uses SSM to
orchestrate the build). The CNG UserData therefore **reinstalls-if-missing + enables** the
agent at first boot (Rocky/RHEL only; no-op on Ubuntu). Confirm the login node is reachable:

```bash
CID=$(aws pcs list-clusters --region $REGION --query "clusters[?name=='pcs-rocky9-cluster'].id | [0]" --output text)
# node groups + queues should all be ACTIVE (authoritative registration signal):
aws pcs list-compute-node-groups --cluster-identifier "$CID" --region $REGION --query 'computeNodeGroups[].{n:name,s:status}' --output table
LOGIN=$(aws ec2 describe-instances --region $REGION \
  --filters "Name=tag:pcs-cluster-id,Values=$CID" "Name=tag:Name,Values=PCS-login" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
aws ssm start-session --target "$LOGIN" --region $REGION    # SSM works because the agent self-installs at boot
```

> On Rocky the interactive user is **`rocky`** (not `ubuntu`). `sudo su - rocky` on the login node.

## Verifying the AMI is PCS-Ready

Launch an instance from the built AMI (or check a booted compute node) and confirm the
contract the rest of the repo depends on:

```bash
cat /opt/aws/pcs/version                                  # PCS agent installed
cat /opt/aws/pcs/scheduler/slurm-25.11/version            # Slurm at the PCS path
ls  /opt/aws/pcs/scheduler/slurm-25.11/{bin,include,lib/slurm}
systemctl cat slurmd >/dev/null && echo "slurmd unit present"
nvidia-smi -L                                             # GPUs (on a GPU instance)
/opt/amazon/efa/bin/fi_info -p efa                        # EFA fabric
enroot version && lfs --version                           # container runtime + Lustre client
```

Then deploy with this `AmiId` and confirm the node groups **register** — go `ACTIVE` in the
PCS API (`aws pcs list-compute-node-groups ...`), which is the authoritative registration
signal. (`sinfo`/`squeue` from the login node also work once slurmd has its `--conf-server` —
see the slurmd `--conf-server` caveat below, now fixed.)

## Caveats and known follow-ups

- **This is a from-scratch build** with many moving version pins (PCS agent, Slurm, NVIDIA,
  EFA, Lustre client). Expect to iterate on component failures the first time — kmod-vs-kernel
  and SELinux are the usual culprits.
- **`glibc-devel` dependency error on `groupinstall "Development Tools"`** —
  `nothing provides glibc = 2.34-N.el9_8 needed by glibc-devel-...from appstream`. This is a
  **BaseOS/AppStream metadata skew**, not a broken group: `glibc` lives in BaseOS and
  `glibc-devel` in AppStream, and the devel pins the exact glibc version-release. When the
  base AMI's stale dnf cache leaves BaseOS behind while AppStream is fresh, the newest
  `glibc-devel` can't find its matching `glibc` and resolution dead-ends (the retry loop can't
  help — it's a deterministic resolution failure, not a network flake). The kernel/toolchain
  component now defends against this with `dnf clean all` + `makecache`, `dnf -y --refresh
  update` (so both repos advance together), and `--nobest` on the group install (falls back to
  the older devel that matches the installed glibc). If you hit it on an older AMI or a manual
  run: `sudo dnf clean all && sudo dnf -y --refresh update && sudo dnf -y --nobest groupinstall
  "Development Tools"`.
- **SELinux** defaults to permissive (see above); tightening to enforcing is a deliberate
  follow-up.
- **Slurm install path:** AWS docs show both `/opt/aws/pcs/scheduler/` (singular) and
  `/opt/aws/pcs/schedulers/` (plural) in places. This repo uses the **singular** path; the
  Slurm component symlinks plural → singular if the installer used the other form.
- **GPU job execution** still requires enough On-Demand **G/VT vCPU quota** for the instance
  type — separate from the AMI/registration path. (Not yet exercised end-to-end; the account
  used for validation had a 64-vCPU G/VT quota, below the 192 vCPU a single .48xlarge needs.)
- **SSM agent is stripped by Image Builder.** The build installs `amazon-ssm-agent`, but EC2
  Image Builder removes it from the output AMI on cleanup (it uses SSM to orchestrate the
  build). The fix lives in the **CNG UserData**, which reinstalls-if-missing + enables the
  agent at first boot (Rocky/RHEL only). This is why nodes are SSM-reachable despite the AMI
  itself shipping without a running agent. **Verified** on a live `pcs-rocky9-cluster` deploy
  (us-east-2, login node on Rocky 9.8): the agent self-installed within ~2 min of boot and the
  node registered `Online` (`amazon-ssm-agent-3.3.4851.0`), reachable via `aws ssm
  start-session` / `send-command`.
- **AWS CLI is absent on stock Rocky and must be baked in.** Unlike the Ubuntu DLAMI, Rocky 9
  ships no `aws` CLI, so the first-boot scripts that use it — post-install/Enroot-Pyxis fetch
  (`aws s3 cp`), GPU health-check tarball fetch (`aws s3 cp`), monitoring Grafana secret
  (`aws ssm put-parameter`) — failed `aws: command not found` (exit 127) on early AMIs.
  Fixed two ways: the build now installs `awscli2` (kernel/toolchain component), **and** the
  CNG UserData installs it if missing at first boot so pre-v1.3.0 AMIs self-heal. **Verified**
  on the live deploy: with `aws` present the post-install chain runs to exit 0 (enroot 3.5.0,
  Pyxis wired into `plugstack.conf.d`). Bake-in requires a **rebuild** (bump `SemanticVersion`).
- **slurmd `--conf-server` vs a pre-baked `/etc/sysconfig/slurmd` (FIXED — was the DNS-SRV
  follow-up, and it was worse than "sinfo doesn't work").** The Enroot/Pyxis step used to write
  the Slurm-bin PATH into `/etc/sysconfig/slurmd`. But the PCS agent's first-boot
  `pcs_bootstrap_config_per_instance.sh` writes that same file with
  `SLURMD_OPTIONS='--conf-server=<ctl>:6817 ...'` **only `if [ ! -f ]`** — so pre-creating it
  made PCS skip the controller endpoint. slurmd then fell back to a DNS-SRV controller lookup
  that fails on Rocky (`resolve_ctls_from_dns_srv: Unknown host`), **never started, and PCS
  health-replaced the node** (seen as a ~30-min login-node replacement — not just a broken
  `sinfo`). Fix: set the PATH via a **systemd drop-in** (`slurmd.service.d/10-slurm-path.conf`)
  instead, leaving `/etc/sysconfig/slurmd` for the PCS agent — in the AMI build, the
  `install-enroot-pyxis.sh` RHEL branch, **and** a CNG-UserData self-heal so pre-v1.3.0 AMIs
  recover at first boot (runs after the PCS bootcmd, before `pcs_bootstrap_finalize.sh` starts
  slurmd). **Verified live:** slurmd goes `active` with `--conf-server`, and `sinfo -N` from the
  Rocky login node lists all queues — so this also closes the old `sinfo`/DNS-SRV follow-up.
- **Interactive user is `rocky`, not `ubuntu`.** Any docs/scripts/UserData that assume the
  `ubuntu` user need the OS-aware branch (the boot scripts already detect this). Spot-check
  when adding new boot-time logic.
- The multi-user directory (`DirectoryService=OpenLDAP-LoginNode`) Rocky path uses
  `openldap-servers` + `authselect` + SELinux and is more involved than the Ubuntu path;
  validate it on a live Rocky login node before relying on it in production.
- **Always bump `SemanticVersion` on every rebuild.** Reusing a version makes EC2 Image
  Builder reuse the *cached component build*, so template edits silently don't take effect.
