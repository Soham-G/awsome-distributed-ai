#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=p2p-bandwidth   # name of your job
#SBATCH --nodes=1                  # intra-node test: a single GPU node
#SBATCH --exclusive                # whole node, so nothing else perturbs the GPUs
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err

# Benchmark 1 — raw GPU peer-to-peer (NVLink/NVSwitch) bandwidth & latency.
#
# Builds and runs NVIDIA's official p2pBandwidthLatencyTest (from NVIDIA/cuda-samples),
# which prints the full pairwise matrices:
#   * Unidirectional bandwidth  (GB/s), P2P disabled vs enabled
#   * Bidirectional  bandwidth  (GB/s), P2P disabled vs enabled
#   * P2P latency               (us),   P2P disabled vs enabled
# This is the cudaMemcpyPeer reference number for every GPU pair on the node.
#
# Runtime: DLAMI-native (no container) — uses the node's own nvcc + driver.
# Submit (partition is NOT hardcoded so this works on any GPU queue):
#   sbatch -p gpu-p5en-spot p2p-bandwidth.sbatch

set -euo pipefail

###########################
###### User Variables #####
###########################

# Where the cuda-samples clone and the compiled binary are cached (shared FS so a
# re-run on any node reuses them). Override with: BUILD_ROOT=/path sbatch ...
: "${BUILD_ROOT:=/fsx/p2p-bandwidth}"
# Set FORCE_REBUILD=1 to recompile even if a cached binary exists.
: "${FORCE_REBUILD:=0}"
# cuda-samples ref to build from (a tag keeps results reproducible).
: "${CUDA_SAMPLES_REF:=master}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

BIN="${BUILD_ROOT}/bin/p2pBandwidthLatencyTest"
SRC_ROOT="${BUILD_ROOT}/cuda-samples"

###########################
####### Preflight #########
###########################

echo "==================== Environment ===================="
echo "Date           : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host           : $(hostname)"
echo "Instance ID    : $(cat /sys/devices/virtual/dmi/id/board_asset_tag 2>/dev/null | tr -d ' ' || echo unknown)"
echo "Slurm job/part : ${SLURM_JOB_ID:-NA} / ${SLURM_JOB_PARTITION:-NA}"
echo "BUILD_ROOT     : ${BUILD_ROOT}"

command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found"; exit 1; }
command -v nvcc       >/dev/null 2>&1 || { echo "ERROR: nvcc not found (looked in ${CUDA_HOME}/bin)"; exit 1; }
echo "nvcc           : $(nvcc --version | tail -1)"

###########################
##### GPU / topology ######
###########################

echo
echo "==================== GPUs ===================="
nvidia-smi -L
echo
echo "==================== NVLink topology (nvidia-smi topo -m) ===================="
# Off-diagonal entries should read NV# (NVLink). PIX/PXB/SYS would mean a GPU pair is
# only reachable over PCIe/host — interpret any low bandwidth pair against this matrix.
nvidia-smi topo -m || true
echo
echo "==================== NVLink status ===================="
nvidia-smi nvlink --status 2>/dev/null || echo "(nvidia-smi nvlink --status unavailable)"

###########################
######### Build ###########
###########################

mkdir -p "${BUILD_ROOT}/bin"

if [[ "${FORCE_REBUILD}" != "1" && -x "${BIN}" ]]; then
    echo
    echo "Reusing cached binary: ${BIN} (set FORCE_REBUILD=1 to rebuild)"
else
    echo
    echo "==================== Build p2pBandwidthLatencyTest ===================="

    # Auto-detect the GPU architecture so this script also works on g7e / p6 / B200.
    # H200 -> 90 -> sm_90 ; B200 -> 100 -> sm_100.
    CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    if [[ -z "${CC_RAW}" ]]; then
        echo "ERROR: could not detect GPU compute capability"; exit 1
    fi
    ARCH="sm_${CC_RAW}"
    echo "Detected compute capability ${CC_RAW} -> -arch=${ARCH}"

    if [[ ! -d "${SRC_ROOT}/.git" ]]; then
        echo "Cloning NVIDIA/cuda-samples (${CUDA_SAMPLES_REF}) into ${SRC_ROOT} ..."
        rm -rf "${SRC_ROOT}"
        git clone --depth 1 --branch "${CUDA_SAMPLES_REF}" \
            https://github.com/NVIDIA/cuda-samples.git "${SRC_ROOT}"
    else
        echo "Reusing existing clone at ${SRC_ROOT}"
    fi

    # Locate the sample source and the header-only Common/ dir (layout has moved across
    # cuda-samples releases — find it rather than hardcoding the path).
    SAMPLE_CU="$(find "${SRC_ROOT}" -name 'p2pBandwidthLatencyTest.cu' 2>/dev/null | head -1)"
    [[ -n "${SAMPLE_CU}" ]] || { echo "ERROR: p2pBandwidthLatencyTest.cu not found in clone"; exit 1; }
    COMMON_DIR="$(dirname "$(find "${SRC_ROOT}" -path '*/Common/helper_cuda.h' 2>/dev/null | head -1)")"
    [[ -n "${COMMON_DIR}" ]] || { echo "ERROR: cuda-samples Common/ headers not found"; exit 1; }
    echo "Source : ${SAMPLE_CU}"
    echo "Headers: ${COMMON_DIR}"

    # The sample is single-file and only needs the header-only helpers, so compile it
    # directly — this avoids the Make->CMake build-system churn between sample releases.
    echo "Compiling ..."
    nvcc -O3 -std=c++14 -arch="${ARCH}" -I"${COMMON_DIR}" -o "${BIN}" "${SAMPLE_CU}"
    echo "Built ${BIN}"
fi

###########################
########## Run ############
###########################

echo
echo "==================== p2pBandwidthLatencyTest ===================="
# Single-node job: this script body already runs on the allocated GPU node, so we run
# the binary directly (no inner srun needed). It enumerates all visible GPUs itself.
"${BIN}"

echo
echo "==================== How to read this ===================="
cat <<'EOF'
* "Unidirectional/Bidirectional Bandwidth Matrix (P2P Enabled)" is the result that
  matters: every off-diagonal cell is GPU[i]->GPU[j] over NVLink/NVSwitch.
* On p5en (8x H200, NVSwitch all-to-all) expect bidirectional P2P-enabled ~700-900 GB/s
  for every pair, and P2P Enabled >> P2P Disabled.
* A pair that sits near PCIe speeds (tens of GB/s) while "topo -m" above showed NV# for
  that pair indicates a degraded link — cross-check with `nvidia-smi nvlink -e`.
* Compare these cells against the OpenMPI 5 osu_bw / osu_bibw numbers from
  osu-p2p-mpi.sbatch (CUDA-aware MPI over the same links).
EOF
