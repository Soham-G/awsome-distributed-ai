#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=osu-internode-efa   # name of your job
#SBATCH --nodes=2                       # INTER-node test: one rank on each of 2 nodes
#SBATCH --ntasks-per-node=1             # 1 rank/node -> traffic crosses the network (EFA)
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err

# Inter-node OSU point-to-point bandwidth over EFA — the across-the-network counterpart
# to the intra-node NVLink tests (osu-p2p-mpi*.sbatch).
#
# Runs osu_bw / osu_bibw between two ranks on two DIFFERENT nodes, with FI_PROVIDER=efa
# so the transport is the Elastic Fabric Adapter. Two buffer modes:
#   BUFFER_TYPE="D D" (default) — DEVICE buffers: GPU<->GPU across nodes. With GPUDirect
#                                 RDMA the NIC DMAs straight from/to GPU memory over EFA.
#   BUFFER_TYPE="H H"           — HOST buffers: CPU<->CPU across nodes (plain EFA).
#
# Uses OpenMPI 5 + the OSU build cached on /fsx by the intra-node scripts (same
# BUILD_ROOT). Runtime: DLAMI-native (no container).
#
# Submit (needs 2 nodes available):
#   sbatch -p gpu-p5en-spot osu-internode-efa.sbatch                                  # D D
#   BUFFER_TYPE="H H" sbatch --export=ALL,BUFFER_TYPE="H H" -p gpu-p5en-spot osu-internode-efa.sbatch

set -euo pipefail

###########################
###### User Variables #####
###########################

: "${BUILD_ROOT:=/fsx/p2p-bandwidth}"          # shared-FS cache (same as intra-node OSU)
: "${FORCE_REBUILD:=0}"
: "${OSU_VERSION:=7.5}"
: "${MAX_BYTES:=268435456}"                      # 256 MiB sweep (EFA needs large msgs to peak)
: "${BUFFER_TYPE:=D D}"                          # "D D" device (GPUDirect RDMA) / "H H" host
# Inter-node MPI transport: cm-ofi (default, PML cm + MTL ofi) or ob1-ofi (PML ob1 + BTL
# ofi). Both ride EFA via libfabric. UCX is NOT available for inter-node here (EFA is not
# a verbs device), so only the two OFI paths are valid across nodes.
: "${MPI_TRANSPORT:=cm-ofi}"

OMPI="${OMPI:-/opt/amazon/openmpi5}"
# CUDA 12 toolkit (CUDA 13 broke OSU's cudaMemPrefetchAsync; see osu-p2p-mpi.sbatch).
if [[ -z "${CUDA_HOME:-}" ]]; then
    for c in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12; do
        [[ -x "${c}/bin/nvcc" ]] && { CUDA_HOME="${c}"; break; }
    done
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export PATH="${OMPI}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${OMPI}/lib:${CUDA_HOME}/lib64:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
export HOME="${HOME:-/tmp}"

# Transport MCA flags (cm-ofi = default auto-select; ob1-ofi = force PML ob1 + BTL ofi).
MCA_ARGS=()
case "${MPI_TRANSPORT}" in
    cm-ofi)  ;;                                                    # default, no override
    ob1-ofi) MCA_ARGS=(--mca pml ob1 --mca btl ofi,self) ;;
    *) echo "ERROR: inter-node supports cm-ofi|ob1-ofi only (got '${MPI_TRANSPORT}')"; exit 1 ;;
esac

# EFA transport.
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1                  # GPUDirect RDMA path for device buffers
export FI_EFA_FORK_SAFE=1

OSU_PREFIX="${BUILD_ROOT}/osu-install-${OSU_VERSION}"
OSU_SRC="osu-micro-benchmarks-${OSU_VERSION}"
OSU_TGZ_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/${OSU_SRC}.tar.gz"

###########################
####### Preflight #########
###########################

echo "==================== Environment ===================="
echo "Date       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes      : ${SLURM_JOB_NUM_NODES:-?} (${SLURM_JOB_NODELIST:-?})"
echo "Slurm job  : ${SLURM_JOB_ID:-NA} / ${SLURM_JOB_PARTITION:-NA}"
echo "OpenMPI    : ${OMPI}  |  CUDA: ${CUDA_HOME}"
echo "Buffers    : ${BUFFER_TYPE}  (D=device/GPUDirect-RDMA, H=host)"
echo "FI_PROVIDER: ${FI_PROVIDER}  FI_EFA_USE_DEVICE_RDMA=${FI_EFA_USE_DEVICE_RDMA}"
command -v mpirun >/dev/null || { echo "ERROR: mpirun not found"; exit 1; }
echo "mpirun     : $(mpirun --version 2>/dev/null | head -1)"

# --- Placement annotation (self-documents the .out file) ---
# Spot nodes launch WITHOUT a cluster placement group (the add-cng templates create a PG
# only for On-Demand). EFA still works without one, but inter-node latency can be higher
# and bandwidth more variable than a placement-grouped pair. Record each node's hostname,
# lifecycle (spot/on-demand) and placement group so the result carries the caveat.
echo "-------------------- node placement annotation --------------------"
srun -l bash -c '
  T=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null)
  H="X-aws-ec2-metadata-token: $T"
  PG=$(curl -sf -H "$H" http://169.254.169.254/latest/meta-data/placement/group-name 2>/dev/null) || PG=""
  [ -z "$PG" ] && PG="(none)"
  LC=$(curl -sf -H "$H" http://169.254.169.254/latest/meta-data/instance-life-cycle 2>/dev/null) || LC="?"
  AZ=$(curl -sf -H "$H" http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null) || AZ="?"
  echo "node=$(hostname) lifecycle=$LC placement_group=$PG az=$AZ"
' || true
echo "ANNOTATION: inter-node EFA P2P. If placement_group=(none) above (the Spot default),"
echo "inter-node latency may be higher / bandwidth more variable than a placement-grouped"
echo "On-Demand pair. EFA itself does not require a placement group."

###########################
######### Build ###########
###########################

find_osu_bin () { find "${OSU_PREFIX}" -name "$1" -type f 2>/dev/null | head -1; }
OSU_BW="$(find_osu_bin osu_bw || true)"
OSU_BIBW="$(find_osu_bin osu_bibw || true)"

if [[ "${FORCE_REBUILD}" == "1" || ! -x "${OSU_BW:-/nonexistent}" || ! -x "${OSU_BIBW:-/nonexistent}" ]]; then
    echo
    echo "==================== Build OSU ${OSU_VERSION} (CUDA) ===================="
    mkdir -p "${BUILD_ROOT}"
    BUILD_TMP="${BUILD_ROOT}/build-${OSU_SRC}"; rm -rf "${BUILD_TMP}"; mkdir -p "${BUILD_TMP}"
    pushd "${BUILD_TMP}" >/dev/null
    curl -fSL -o osu.tgz "${OSU_TGZ_URL}"; tar xf osu.tgz; cd "${OSU_SRC}"
    ./configure CC="$(command -v mpicc)" CXX="$(command -v mpicxx)" \
        --enable-cuda --with-cuda="${CUDA_HOME}" --prefix="${OSU_PREFIX}"
    make -j"$(nproc)"; make install
    popd >/dev/null
    OSU_BW="$(find_osu_bin osu_bw)"; OSU_BIBW="$(find_osu_bin osu_bibw)"
else
    echo "Reusing cached OSU build under ${OSU_PREFIX}"
fi
echo "osu_bw   : ${OSU_BW}"
echo "osu_bibw : ${OSU_BIBW}"

###########################
########## Run ############
###########################

read -r BUF_SEND BUF_RECV <<< "${BUFFER_TYPE}"
DEV_ARGS=()
[[ "${BUF_SEND}" == "D" || "${BUF_RECV}" == "D" ]] && DEV_ARGS=(-d cuda)

run () {
    local bin="$1" label="$2"
    echo
    echo "-------------------- ${label} [${BUF_SEND} ${BUF_RECV}] [${MPI_TRANSPORT}] over EFA --------------------"
    # 2 ranks, one per node (PCS/Slurm places them); each rank uses GPU 0 on its node.
    # MCA_ARGS forces the transport (empty for cm-ofi default).
    mpirun -np 2 --map-by ppr:1:node --bind-to none \
        "${MCA_ARGS[@]}" \
        -x PATH -x LD_LIBRARY_PATH -x HOME \
        -x FI_PROVIDER -x FI_EFA_USE_DEVICE_RDMA -x FI_EFA_FORK_SAFE \
        -x CUDA_VISIBLE_DEVICES=0 \
        "${bin}" "${DEV_ARGS[@]}" -m "${MAX_BYTES}" "${BUF_SEND}" "${BUF_RECV}"
}

echo
echo "==================== OSU inter-node P2P over EFA (${BUFFER_TYPE}) ===================="
status=0
run "${OSU_BW}"   "osu_bw  (unidirectional)" || status=$?
run "${OSU_BIBW}" "osu_bibw (bidirectional)" || status=$?

echo
if [[ "${status}" -eq 0 ]]; then
    echo "RESULT: PASS — inter-node OSU (${BUFFER_TYPE}) over EFA completed."
else
    echo "RESULT: FAIL — a run errored (exit ${status}). Check FI_PROVIDER=efa selected and,"
    echo "for D D, that GPUDirect RDMA (nvidia-peermem / FI_EFA_USE_DEVICE_RDMA) is available."
fi

echo
echo "==================== How to read this ===================="
cat <<'EOF'
* This is INTER-node bandwidth over EFA — compare it against the intra-node NVLink
  numbers (osu-p2p-mpi*.sbatch). EFA peak per interface is far below NVLink, so expect
  the inter-node curve to plateau well under the intra-node one.
* D D (device) uses GPUDirect RDMA: the EFA NIC reads/writes GPU memory directly.
  H H (host) is plain CPU<->CPU EFA. Comparing them shows the GPUDirect benefit.
* Peak depends on how many EFA NICs a single MPI pair drives (one pair = one rail);
  multi-rail aggregate needs multiple pairs (osu_mbw_mr) — this test is single-pair.
* GB/s base-10, comparable with the other benchmarks here.
EOF
exit "${status}"
