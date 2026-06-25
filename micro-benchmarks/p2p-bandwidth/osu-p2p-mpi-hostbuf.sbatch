#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=osu-p2p-mpi-hostbuf   # name of your job
#SBATCH --nodes=1                        # intra-node test: both ranks on one node
#SBATCH --ntasks=2                       # 2 ranks
#SBATCH --exclusive                      # whole node, so nothing else perturbs the run
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err

# Benchmark 2b — NON-CUDA-aware (HOST buffer) MPI point-to-point bandwidth BASELINE.
#
# Companion to osu-p2p-mpi.sbatch (the CUDA-aware "D D" device-buffer test). This one runs
# the SAME osu_bw / osu_bibw with HOST buffers ("H H"): both send and receive buffers live
# in CPU memory, so the data path is CPU<->CPU shared memory — NOT NVLink. It is the
# baseline you compare the CUDA-aware device numbers against to see what CUDA awareness
# actually buys (divide the D-D bandwidth by these H-H numbers).
#
# It deliberately mirrors osu-p2p-mpi.sbatch (same OpenMPI 5, same OSU build cached on
# /fsx) and reuses the cached binaries, so running both side by side reproduces the full
# comparison. Kept as a separate file so each script reproduces exactly one result.
#
# NOTE: "H H" never touches the GPUs, so there is no GPU pinning and no GPU pair loop —
# the result is a single host shared-memory bandwidth curve between two ranks on the node.
#
# Runtime: DLAMI-native (no container). Submit:
#   sbatch -p gpu-p5en-spot osu-p2p-mpi-hostbuf.sbatch

set -euo pipefail

###########################
###### User Variables #####
###########################

: "${BUILD_ROOT:=/fsx/p2p-bandwidth}"          # shared-FS cache (same root as the CUDA-aware run)
: "${FORCE_REBUILD:=0}"                          # 1 = rebuild OSU even if cached
: "${OSU_VERSION:=7.5}"                          # OSU Micro-Benchmarks release (match the D D run)
# Max message size in BYTES for the sweep. Empty = OSU default (up to 4 MiB).
# Set MAX_BYTES=268435456 (256 MiB) to match the CUDA-aware sweep.
: "${MAX_BYTES:=}"

OMPI="${OMPI:-/opt/amazon/openmpi5}"             # OpenMPI 5 (NOT the default 4.1.x)

# Build OSU against a CUDA *12* toolkit (same reason as the CUDA-aware script: CUDA 13
# changed the cudaMemPrefetchAsync() signature and OSU <=7.5 won't compile against it).
# Even though this host-buffer run never uses the GPU, we build identically so both
# scripts share one cached OSU install under BUILD_ROOT.
if [[ -z "${CUDA_HOME:-}" ]]; then
    for c in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12; do
        [[ -x "${c}/bin/nvcc" ]] && { CUDA_HOME="${c}"; break; }
    done
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"     # last resort: default symlink
fi
export PATH="${OMPI}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${OMPI}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# OpenMPI 5 errors out ("Unable to get the user home directory") when HOME is unset.
export HOME="${HOME:-/tmp}"

OSU_SRC="osu-micro-benchmarks-${OSU_VERSION}"
OSU_TGZ_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/${OSU_SRC}.tar.gz"
OSU_PREFIX="${BUILD_ROOT}/osu-install-${OSU_VERSION}"

###########################
####### Preflight #########
###########################

echo "==================== Environment ===================="
echo "Date         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host         : $(hostname)"
echo "Instance ID  : $(cat /sys/devices/virtual/dmi/id/board_asset_tag 2>/dev/null | tr -d ' ' || echo unknown)"
echo "Slurm job    : ${SLURM_JOB_ID:-NA} / ${SLURM_JOB_PARTITION:-NA}"
echo "OpenMPI      : ${OMPI}"
echo "Mode         : NON-CUDA-aware HOST buffers (H H) — CPU<->CPU baseline (no NVLink)"

command -v mpicc   >/dev/null 2>&1 || { echo "ERROR: mpicc not found under ${OMPI}/bin"; exit 1; }
command -v mpirun  >/dev/null 2>&1 || { echo "ERROR: mpirun not found under ${OMPI}/bin"; exit 1; }
command -v nvcc    >/dev/null 2>&1 || { echo "ERROR: nvcc not found (looked in ${CUDA_HOME}/bin)"; exit 1; }
echo "CUDA_HOME    : ${CUDA_HOME}"
echo "mpirun       : $(mpirun --version 2>/dev/null | head -1)"

###########################
######### Build ###########
###########################

# Resolve the pt2pt binaries (install layout: <prefix>/libexec/osu-micro-benchmarks/mpi/pt2pt/)
find_osu_bin () { find "${OSU_PREFIX}" -name "$1" -type f 2>/dev/null | head -1; }

OSU_BW="$(find_osu_bin osu_bw || true)"
OSU_BIBW="$(find_osu_bin osu_bibw || true)"

if [[ "${FORCE_REBUILD}" != "1" && -x "${OSU_BW:-/nonexistent}" && -x "${OSU_BIBW:-/nonexistent}" ]]; then
    echo
    echo "Reusing cached OSU build under ${OSU_PREFIX} (set FORCE_REBUILD=1 to rebuild)"
else
    echo
    echo "==================== Build OSU Micro-Benchmarks ${OSU_VERSION} (CUDA) ===================="
    mkdir -p "${BUILD_ROOT}"
    BUILD_TMP="${BUILD_ROOT}/build-${OSU_SRC}"
    rm -rf "${BUILD_TMP}"; mkdir -p "${BUILD_TMP}"
    pushd "${BUILD_TMP}" >/dev/null

    echo "Downloading ${OSU_TGZ_URL} ..."
    curl -fSL -o osu.tgz "${OSU_TGZ_URL}"
    tar xf osu.tgz
    cd "${OSU_SRC}"

    echo "Configuring with CUDA support ..."
    ./configure CC="$(command -v mpicc)" CXX="$(command -v mpicxx)" \
        --enable-cuda --with-cuda="${CUDA_HOME}" \
        --prefix="${OSU_PREFIX}"
    echo "Building ..."
    make -j"$(nproc)"
    make install
    popd >/dev/null

    OSU_BW="$(find_osu_bin osu_bw)"
    OSU_BIBW="$(find_osu_bin osu_bibw)"
fi

[[ -x "${OSU_BW:-/nonexistent}"   ]] || { echo "ERROR: osu_bw not found after build";   exit 1; }
[[ -x "${OSU_BIBW:-/nonexistent}" ]] || { echo "ERROR: osu_bibw not found after build"; exit 1; }
echo "osu_bw   : ${OSU_BW}"
echo "osu_bibw : ${OSU_BIBW}"

###########################
########## Run ############
###########################

# Optional max-message-size flag for the sweep (empty = OSU default).
OSU_MAX_ARGS=()
[[ -n "${MAX_BYTES}" ]] && OSU_MAX_ARGS=(-m "${MAX_BYTES}")

run_host () {
    local bin="$1" label="$2"
    echo
    echo "-------------------- ${label} [H H] --------------------"
    # 2 ranks on this one node, host buffers. Bind to cores so the two ranks land on
    # separate cores. No -d cuda, no CUDA_VISIBLE_DEVICES — pure CPU-memory transfer.
    mpirun -np 2 --map-by ppr:2:node --bind-to core \
        -x PATH -x LD_LIBRARY_PATH -x HOME \
        "${bin}" H H "${OSU_MAX_ARGS[@]}"
}

echo
echo "==================== OpenMPI 5 OSU P2P: NON-CUDA-aware HOST buffers (H H) ===================="
status=0
run_host "${OSU_BW}"   "osu_bw  (unidirectional, host buffers)" || status=$?
run_host "${OSU_BIBW}" "osu_bibw (bidirectional, host buffers)" || status=$?

echo
if [[ "${status}" -eq 0 ]]; then
    echo "RESULT: PASS — host-buffer (H H) baseline completed on OpenMPI 5."
else
    echo "RESULT: FAIL — a host-buffer run errored (exit ${status})."
fi

echo
echo "==================== How to read this ===================="
cat <<'EOF'
* This is the NON-CUDA-aware baseline: both buffers are in CPU memory, so the transfer
  is CPU<->CPU host shared-memory bandwidth on the node — it does NOT use the GPUs or
  NVLink at all (which is why there is no GPU pair loop here).
* Compare against the CUDA-aware device-buffer ("D D") numbers from osu-p2p-mpi.sbatch:
    speedup = (D-D bandwidth) / (H-H bandwidth)
  On a p5en the device path rides NVLink (~hundreds of GB/s) while this host path is
  bounded by CPU memory / intra-node shared-memory copy — the ratio is what CUDA-aware
  MPI buys for GPU<->GPU traffic.
* osu_bw = peak unidirectional GB/s; osu_bibw = bidirectional GB/s; both base-10 (1e9 B/s).
EOF
exit "${status}"
