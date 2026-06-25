#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=nccl-internode-efa   # name of your job
#SBATCH --nodes=2                        # INTER-node: traffic crosses EFA
#SBATCH --ntasks-per-node=1              # 1 rank (1 GPU) per node for a clean P2P pair
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err

# Inter-node NCCL point-to-point bandwidth over EFA — the across-the-network counterpart
# to the intra-node NVLink NCCL test (nccl-sendrecv.sbatch).
#
# Runs nccl-tests `sendrecv_perf` between 1 GPU on each of 2 nodes, sweeping message
# sizes. NCCL selects the AWS OFI/EFA transport (NET/OFI provider efa) for inter-node,
# using GPUDirect RDMA where available — this is the path real multi-node training uses.
#
# Uses the prebuilt nccl-tests CONTAINER via Pyxis (same image as nccl-sendrecv.sbatch).
#
# Submit (needs 2 nodes available):
#   sbatch -p gpu-p5en-spot nccl-internode-efa.sbatch

set -euo pipefail

###########################
###### User Variables #####
###########################

: "${APPS_PATH:=/fsx}"
: "${NCCL_TESTS_TAG:=cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9}"
: "${IMAGE:=public.ecr.aws/hpc-cloud/nccl-tests:${NCCL_TESTS_TAG}}"
: "${SQSH:=${APPS_PATH}/nccl-tests-sendrecv.sqsh}"   # shared with intra-node script
: "${BEGIN_BYTES:=8}"
: "${END_BYTES:=268435456}"                          # 256 MiB
: "${STEP_FACTOR:=2}"
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"

###########################
####### Preflight #########
###########################

echo "==================== Environment ===================="
echo "Date     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes    : ${SLURM_JOB_NUM_NODES:-?} (${SLURM_JOB_NODELIST:-?})"
echo "Slurm    : ${SLURM_JOB_ID:-NA} / ${SLURM_JOB_PARTITION:-NA}"
echo "Image    : ${IMAGE}"
echo "Sweep    : ${BEGIN_BYTES}..${END_BYTES} bytes (x${STEP_FACTOR})"
srun -l hostname || true   # show the two distinct nodes

# EFA transport for NCCL inter-node. NCCL_P2P intra-node is irrelevant here (1 GPU/node);
# the inter-node path goes through the AWS OFI NCCL plugin over EFA.
export NCCL_DEBUG=INFO
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
# Keep NCCL off the docker/loopback interfaces for its bootstrap.
export NCCL_SOCKET_IFNAME=^docker,lo,veth

###########################
###### Import image #######
###########################

if [[ ! -f "${SQSH}" ]]; then
    echo "Importing ${IMAGE} -> ${SQSH} ..."
    enroot import -o "${SQSH}" "docker://${IMAGE/public.ecr.aws\//public.ecr.aws#}"
else
    echo "Reusing cached image ${SQSH}"
fi

###########################
########## Run ############
###########################

echo
echo "==================== NCCL sendrecv_perf inter-node over EFA (2 nodes x 1 GPU) ===================="
srun --mpi=pmix --cpu-bind=none \
     --container-image="${SQSH}" \
     -x FI_PROVIDER -x FI_EFA_USE_DEVICE_RDMA -x FI_EFA_FORK_SAFE \
     -x NCCL_DEBUG -x NCCL_SOCKET_IFNAME \
     "${NCCL_TESTS_PATH}/sendrecv_perf" \
        -b "${BEGIN_BYTES}" -e "${END_BYTES}" -f "${STEP_FACTOR}" -g 1 -c 1 -n 100
status=$?

echo
if [[ "${status}" -eq 0 ]]; then
    echo "RESULT: PASS — inter-node NCCL sendrecv_perf over EFA completed."
    echo "Confirm EFA was used: look for 'NET/OFI Selected provider is efa' in the log above."
else
    echo "RESULT: FAIL — NCCL run errored (exit ${status})."
fi

echo
echo "==================== How to read this ===================="
cat <<'EOF'
* busbw (GB/s) per message size is the inter-node P2P bandwidth over EFA.
* Compare against:
    - nccl-sendrecv.sbatch (intra-node NCCL over NVLink) — same tool, vastly higher BW
    - osu-internode-efa.sbatch (MPI over EFA) — same wire, different library
* EFA is confirmed active when the NCCL_DEBUG log shows
  "NET/OFI Selected provider is efa ... (found N nics)". If it falls back to TCP/sockets,
  busbw will be far lower — check the AWS OFI NCCL plugin + EFA in the container/AMI.
* A single 1-GPU-per-node pair drives a subset of the node's EFA bandwidth; full
  multi-rail aggregate needs all 8 GPUs (all_reduce_perf -g 8 across 2 nodes).
EOF
exit "${status}"
