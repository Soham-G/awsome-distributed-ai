#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=nccl-sendrecv   # name of your job
#SBATCH --nodes=1                  # intra-node test: both ranks on one GPU node
#SBATCH --ntasks-per-node=2        # 2 ranks -> 2 GPUs, point-to-point
#SBATCH --gpus-per-node=2          # only need 2 of the 8 GPUs
#SBATCH --exclusive                # whole node so nothing else perturbs the GPUs
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err

# Benchmark 3 — NCCL point-to-point bandwidth across a message-size SWEEP.
#
# Runs NVIDIA's nccl-tests `sendrecv_perf` between 2 GPUs on a single node, sweeping
# message sizes (-b .. -e, doubling). This is the NCCL analog of the OSU pt2pt tests
# and the closest collective-library comparison to the raw cudaMemcpyPeer matrix: it
# rides the same NVLink/NVSwitch fabric, but through NCCL (what PyTorch/Megatron use).
#
# Unlike the other three benchmarks here (DLAMI-native), this one uses the prebuilt
# nccl-tests CONTAINER via Pyxis (the repo's documented path, see ../nccl-tests/),
# because nccl-tests + a matching NCCL/EFA stack are already baked into the image.
#
# Submit:
#   sbatch -p gpu-p5en-spot nccl-sendrecv.sbatch

set -euo pipefail

###########################
###### User Variables #####
###########################

: "${APPS_PATH:=/fsx}"                                   # shared FS for the .sqsh image
# Pinned nccl-tests image tag (same family the repo's nccl-tests README uses).
: "${NCCL_TESTS_TAG:=cuda12.8.1-efa1.43.2-ofiv1.16.3-ncclv2.27.7-1-testsv2.16.9}"
: "${IMAGE:=public.ecr.aws/hpc-cloud/nccl-tests:${NCCL_TESTS_TAG}}"
: "${SQSH:=${APPS_PATH}/nccl-tests-sendrecv.sqsh}"
# Sweep: begin / end / factor. 8 B .. 256 MiB, doubling — matches the OSU sweep range.
: "${BEGIN_BYTES:=8}"
: "${END_BYTES:=268435456}"
: "${STEP_FACTOR:=2}"
# nccl-tests binary path inside the container.
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"

###########################
####### Preflight #########
###########################

echo "==================== Environment ===================="
echo "Date     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host     : $(hostname)"
echo "Instance : $(cat /sys/devices/virtual/dmi/id/board_asset_tag 2>/dev/null | tr -d ' ' || echo unknown)"
echo "Slurm    : ${SLURM_JOB_ID:-NA} / ${SLURM_JOB_PARTITION:-NA}"
echo "Image    : ${IMAGE}"
echo "Sweep    : ${BEGIN_BYTES}..${END_BYTES} bytes (x${STEP_FACTOR})"
nvidia-smi -L 2>/dev/null || true

# Intra-node only: keep NCCL on NVLink/NVSwitch (P2P) and off the network, so this is a
# clean GPU<->GPU P2P number comparable to the OSU/cudaMemcpyPeer pairwise results.
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL        # use NVLink for intra-node P2P
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

###########################
###### Import image #######
###########################

# enroot's overlay needs node-local root disk during import; the resulting .sqsh lands
# on shared /fsx. Reuse a cached .sqsh if present.
if [[ ! -f "${SQSH}" ]]; then
    echo "Importing ${IMAGE} -> ${SQSH} ..."
    enroot import -o "${SQSH}" "docker://${IMAGE/public.ecr.aws\//public.ecr.aws#}"
else
    echo "Reusing cached image ${SQSH}"
fi

###########################
########## Run ############
###########################

# sendrecv_perf with -g 1 means 1 GPU per rank; 2 ranks => a 2-GPU P2P send/recv ring.
# -b/-e/-f sweep the message size; -c 1 validates; -n 100 iterations per size.
echo
echo "==================== NCCL sendrecv_perf (2 GPUs, message-size sweep) ===================="
srun --mpi=pmix --cpu-bind=none \
     --container-image="${SQSH}" \
     "${NCCL_TESTS_PATH}/sendrecv_perf" \
        -b "${BEGIN_BYTES}" -e "${END_BYTES}" -f "${STEP_FACTOR}" -g 1 -c 1 -n 100
status=$?

echo
if [[ "${status}" -eq 0 ]]; then
    echo "RESULT: PASS — NCCL sendrecv_perf sweep completed."
else
    echo "RESULT: FAIL — NCCL run errored (exit ${status})."
    echo "If it failed on the container launch, confirm Pyxis is available on the node"
    echo "(spank_pyxis.so + enroot); these benchmarks otherwise run DLAMI-native."
fi

echo
echo "==================== How to read this ===================="
cat <<'EOF'
* nccl-tests prints, per message size: size, count, ..., algbw (GB/s), busbw (GB/s).
* For sendrecv between 2 GPUs, busbw ~= the achieved P2P bandwidth over NVLink, so the
  busbw column is what to compare against:
    - OSU CUDA-aware (D D) osu_bw       (MPI point-to-point over the same NVLink)
    - NVIDIA cudaMemcpyPeer @160MB peak (raw CUDA, hardware ceiling)
* NCCL typically sits between raw cudaMemcpyPeer and CUDA-aware MPI: it is purpose-built
  for GPU transfers (kernel-driven, NVLink-aware) but still has launch/protocol overhead
  the bare peer-copy doesn't. All report GB/s base-10.
* Look for "NET/OFI" lines in NCCL_DEBUG only at multi-node scale; here NCCL should pick
  the P2P/NVLink transport (no EFA), confirmed by NCCL_P2P_LEVEL=NVL.
EOF
exit "${status}"
