#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=osu-p2p-mpi   # name of your job
#SBATCH --nodes=1                # intra-node test: both ranks on one GPU node
#SBATCH --ntasks=2               # 2 ranks, one per GPU under test
#SBATCH --exclusive              # whole node, so nothing else perturbs the GPUs
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err

# Benchmark 2 — CUDA-aware MPI GPU<->GPU point-to-point bandwidth, for comparison with
# the raw cudaMemcpyPeer numbers from p2p-bandwidth.sbatch (Benchmark 1).
#
# Uses OpenMPI 5 (/opt/amazon/openmpi5) + the OSU Micro-Benchmarks built with CUDA, and
# runs osu_bw (unidirectional) and osu_bibw (bidirectional) with DEVICE buffers ("D D")
# between two ranks each pinned to a different NVLink-connected GPU on the same node.
#
# Runtime: DLAMI-native (no container). Submit:
#   sbatch -p gpu-p5en-spot osu-p2p-mpi.sbatch

set -euo pipefail

###########################
###### User Variables #####
###########################

: "${BUILD_ROOT:=/fsx/p2p-bandwidth}"          # shared-FS cache (same root as Benchmark 1)
: "${FORCE_REBUILD:=0}"                          # 1 = rebuild OSU even if cached
: "${OSU_VERSION:=7.5}"                          # OSU Micro-Benchmarks release
# GPU pairs to test, space-separated "a:b". 0:1 is an NVSwitch-near pair; 0:7 a far pair.
: "${GPU_PAIRS:=0:1 0:7}"
# Max message size in BYTES for the sweep. Empty = OSU's built-in default (up to 4 MiB).
# Set e.g. MAX_BYTES=268435456 (256 MiB) to push past the default into the bandwidth
# plateau. OSU sweeps powers of two from 1 byte up to this max.
: "${MAX_BYTES:=}"

# MPI transport to exercise (which OpenMPI build + MCA path):
#   cm-ofi  (default) — EFA-bundled /opt/amazon/openmpi5, PML cm + MTL ofi (libfabric).
#                       This is what OpenMPI picks by default on EFA (mtl_ofi prio 25 >
#                       ob1 prio 20). Intra-node it still rides NVLink via the OFI shm/CUDA path.
#   ob1-ofi           — same OpenMPI, but force PML ob1 + BTL ofi,smcuda,sm,self. MPI does
#                       the tag-matching; libfabric/shm is just byte transport.
#   ucx               — a SEPARATELY BUILT UCX-enabled OpenMPI (build-ucx-stack.sh). Intra
#                       -node UCX uses the cuda_ipc transport over NVLink. (Intra-node ONLY:
#                       EFA is not a verbs device here, so UCX can't do inter-node EFA.)
: "${MPI_TRANSPORT:=cm-ofi}"

# UCX stack locations (only used when MPI_TRANSPORT=ucx; built by build-ucx-stack.sh).
: "${UCX_VERSION:=1.18.0}"
: "${OMPI_UCX_VERSION:=5.0.6}"

OMPI="${OMPI:-/opt/amazon/openmpi5}"             # default: EFA-bundled OpenMPI 5

# Build OSU against a CUDA *12* toolkit. CUDA 13 changed the cudaMemPrefetchAsync()
# signature (3rd arg is now `struct cudaMemLocation`, not an int device id), which
# OSU <=7.5 does not yet support — `osu_util_mpi.c` fails to compile under CUDA 13.x.
# The H200 driver (R5xx) runs CUDA 12 binaries fine, so we pick the newest CUDA 12
# present on the DLAMI. Override CUDA_HOME to force a specific toolkit.
if [[ -z "${CUDA_HOME:-}" ]]; then
    for c in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12; do
        [[ -x "${c}/bin/nvcc" ]] && { CUDA_HOME="${c}"; break; }
    done
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"     # last resort: default symlink
fi
# --- Transport selection: pick the OpenMPI build, OSU install, and MCA flags ---
MCA_ARGS=()
case "${MPI_TRANSPORT}" in
    cm-ofi)
        # Default EFA OpenMPI; let it auto-select (PML cm + MTL ofi). No MCA override.
        OSU_PREFIX="${BUILD_ROOT}/osu-install-${OSU_VERSION}"
        ;;
    ob1-ofi)
        # Same OpenMPI, force the OB1 + BTL path (ofi for net, smcuda/sm for intra-node).
        MCA_ARGS=(--mca pml ob1 --mca btl ofi,smcuda,sm,self)
        OSU_PREFIX="${BUILD_ROOT}/osu-install-${OSU_VERSION}"
        ;;
    ucx)
        # Separately built UCX-enabled OpenMPI + its own OSU (build-ucx-stack.sh).
        OMPI="${BUILD_ROOT}/openmpi-ucx-${OMPI_UCX_VERSION}"
        OSU_PREFIX="${BUILD_ROOT}/osu-install-${OSU_VERSION}-ucx"
        MCA_ARGS=(--mca pml ucx)
        # OPAL_PREFIX is REQUIRED so this OpenMPI loads its OWN plugins (incl. the ucx
        # PML) and not the system /opt/amazon/openmpi's — without it ompi_info/mpirun
        # resolve against the wrong install and 'pml ucx' is missing.
        export OPAL_PREFIX="${OMPI}"
        export LD_LIBRARY_PATH="${BUILD_ROOT}/ucx-install-${UCX_VERSION}/lib:${LD_LIBRARY_PATH:-}"
        ;;
    *)
        echo "ERROR: unknown MPI_TRANSPORT='${MPI_TRANSPORT}' (use cm-ofi|ob1-ofi|ucx)"; exit 1
        ;;
esac

export PATH="${OMPI}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${OMPI}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# OpenMPI 5 errors out ("Unable to get the user home directory") when HOME is unset.
export HOME="${HOME:-/tmp}"

OSU_SRC="osu-micro-benchmarks-${OSU_VERSION}"
OSU_TGZ_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/${OSU_SRC}.tar.gz"

###########################
####### Preflight #########
###########################

echo "==================== Environment ===================="
echo "Date         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host         : $(hostname)"
echo "Instance ID  : $(cat /sys/devices/virtual/dmi/id/board_asset_tag 2>/dev/null | tr -d ' ' || echo unknown)"
echo "Slurm job    : ${SLURM_JOB_ID:-NA} / ${SLURM_JOB_PARTITION:-NA}"
echo "MPI_TRANSPORT: ${MPI_TRANSPORT}   (mca: ${MCA_ARGS[*]:-<default>})"
echo "OpenMPI      : ${OMPI}"

# For the UCX transport the OpenMPI + OSU must have been pre-built by build-ucx-stack.sh.
if [[ "${MPI_TRANSPORT}" == "ucx" && ! -x "${OMPI}/bin/mpirun" ]]; then
    echo "ERROR: UCX OpenMPI not found at ${OMPI}. Run build-ucx-stack.sh first."; exit 1
fi

command -v mpicc   >/dev/null 2>&1 || { echo "ERROR: mpicc not found under ${OMPI}/bin"; exit 1; }
command -v mpirun  >/dev/null 2>&1 || { echo "ERROR: mpirun not found under ${OMPI}/bin"; exit 1; }
command -v nvcc    >/dev/null 2>&1 || { echo "ERROR: nvcc not found (looked in ${CUDA_HOME}/bin)"; exit 1; }
echo "CUDA_HOME    : ${CUDA_HOME}"
echo "mpirun       : $(mpirun --version 2>/dev/null | head -1)"
echo "nvcc         : $(nvcc --version | grep release)"
echo
echo "==================== CUDA-awareness (ompi_info) ===================="
# Best-effort signal; the real proof is the osu_bw "D D" run completing below.
ompi_info 2>/dev/null | grep -iE 'Open MPI:|cuda' || echo "(no cuda line in ompi_info output)"

###########################
######### Build ###########
###########################

# Resolve the pt2pt binaries (install layout: <prefix>/libexec/osu-micro-benchmarks/mpi/pt2pt/)
find_osu_bin () { find "${OSU_PREFIX}" -name "$1" -type f 2>/dev/null | head -1; }

OSU_BW="$(find_osu_bin osu_bw || true)"
OSU_BIBW="$(find_osu_bin osu_bibw || true)"

if [[ "${MPI_TRANSPORT}" == "ucx" && ( ! -x "${OSU_BW:-/nonexistent}" || ! -x "${OSU_BIBW:-/nonexistent}" ) ]]; then
    echo "ERROR: UCX OSU not found under ${OSU_PREFIX}. Run build-ucx-stack.sh first."; exit 1
fi

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
###### GPU pinning ########
###########################

# Per-rank wrapper pins each rank to a DISTINCT physical GPU so the D D transfer is a
# real GPU<->GPU NVLink copy.
#
# WHY NOT reorder CUDA_VISIBLE_DEVICES per rank: OSU selects its device as
# (local_rank % visible_device_count), NOT "the first visible device". If we expose only
# 2 GPUs, both ranks compute an index into a 2-GPU list and — because we reordered the
# list per rank — BOTH land on the SAME physical GPU. That turns the test into an
# intra-GPU HBM copy (~1.9 TB/s, ~4x over a single H200's ~478 GB/s NVLink ceiling) — a
# bogus "P2P" number. Instead, expose ALL GPUs unchanged and pin each rank explicitly to
# the requested physical GPU via CUDA_VISIBLE_DEVICES = a single device, so OSU's
# local_rank%1 == 0 -> that one device. rank 0 -> gpuA, rank 1 -> gpuB (two real GPUs).
WRAPPER="${BUILD_ROOT}/osu_pair_wrapper.sh"
cat > "${WRAPPER}" <<'WRAP'
#!/bin/bash
# args: <gpuA> <gpuB> <osu_binary> [extra osu OPTION args...]
# IMPORTANT: OSU wants all options BEFORE the "SRC DST" positionals (D D), so we append
# the buffer types LAST, after -d cuda and any extra options like -m <max>.
gpuA="$1"; gpuB="$2"; bin="$3"; shift 3
lr="${OMPI_COMM_WORLD_LOCAL_RANK:-0}"
# Expose exactly ONE physical GPU to each rank: rank 0 sees only gpuA, rank 1 only gpuB.
# With a single visible device, OSU's local_rank%1=0 maps to it — no same-GPU collision.
if [[ "${lr}" == "0" ]]; then export CUDA_VISIBLE_DEVICES="${gpuA}";
else                          export CUDA_VISIBLE_DEVICES="${gpuB}"; fi
exec "${bin}" -d cuda "$@" D D
WRAP
chmod +x "${WRAPPER}"

# Optional max-message-size flag for the sweep (empty = OSU default).
OSU_MAX_ARGS=()
[[ -n "${MAX_BYTES}" ]] && OSU_MAX_ARGS=(-m "${MAX_BYTES}")

run_pair () {
    local a="$1" b="$2" bin="$3" label="$4"
    echo
    echo "-------------------- ${label} [${MPI_TRANSPORT}]: GPU ${a} <-> GPU ${b} --------------------"
    # 2 ranks on this one node; bind to cores to keep the host copy threads off each other.
    # MCA_ARGS forces the transport (empty for cm-ofi default).
    mpirun -np 2 --map-by ppr:2:node --bind-to core \
        "${MCA_ARGS[@]}" \
        -x PATH -x LD_LIBRARY_PATH -x HOME ${OPAL_PREFIX:+-x OPAL_PREFIX} \
        "${WRAPPER}" "${a}" "${b}" "${bin}" "${OSU_MAX_ARGS[@]}"
}

###########################
########## Run ############
###########################

echo
echo "==================== OpenMPI 5 CUDA-aware OSU P2P (device buffers) ===================="
status=0
for pair in ${GPU_PAIRS}; do
    a="${pair%%:*}"; b="${pair##*:}"
    run_pair "${a}" "${b}" "${OSU_BW}"   "osu_bw  (unidirectional, GB unit)" || status=$?
    run_pair "${a}" "${b}" "${OSU_BIBW}" "osu_bibw (bidirectional, GB unit)" || status=$?
done

echo
if [[ "${status}" -eq 0 ]]; then
    echo "RESULT: PASS — CUDA-aware device-buffer (D D) transfers completed on OpenMPI 5."
else
    echo "RESULT: FAIL — a device-buffer run errored (exit ${status})."
    echo "If it failed immediately, this OpenMPI may lack CUDA support; rerun Benchmark 1"
    echo "to confirm the GPUs/NVLink themselves are healthy."
fi

echo
echo "==================== How to read this ===================="
cat <<'EOF'
* osu_bw reports peak unidirectional GB/s; osu_bibw reports bidirectional GB/s.
* Compare each pair against the matching cell of the NVIDIA p2pBandwidthLatencyTest
  matrices (Benchmark 1, p2p-bandwidth.sbatch):
    osu_bw   <-> "Unidirectional Bandwidth Matrix (P2P Enabled)" cell [a][b]
    osu_bibw <-> "Bidirectional  Bandwidth Matrix (P2P Enabled)" cell [a][b]
* CUDA-aware MPI usually lands somewhat below the raw cudaMemcpyPeer ceiling (UCX/MPI
  protocol + staging overhead), but both should be far above PCIe (tens of GB/s).
* OSU "GB/s" here is base-10 (1e9 B/s); the NVIDIA tool also reports GB/s — they are
  directly comparable to within a few %.
EOF
exit "${status}"
