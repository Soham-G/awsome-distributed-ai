#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=fabtests-internode-efa   # name of your job
#SBATCH --nodes=2                            # INTER-node: server on one, client on the other
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err

# Inter-node libfabric fabtests bandwidth over EFA — the RAWEST fabric number, below both
# MPI (osu-internode-efa.sbatch) and NCCL (nccl-internode-efa.sbatch). Those go through
# libfabric/EFA under the hood; this exercises libfabric directly with no MPI or NCCL.
#
# Runs fi_rdm_tagged_bw (the RDM tagged-message bandwidth test) between 2 nodes, in two
# buffer modes, sweeping message sizes:
#   * host buffers  (default)         -> CPU<->CPU over EFA
#   * device buffers (-D cuda, BUF=cuda) -> GPU<->GPU over EFA (GPUDirect RDMA)
#
# fabtests is built from the v2.4.0 tag to match the installed libfabric 2.4.0amzn1
# (test/lib ABI should match), with CUDA enabled, cached on /fsx. DLAMI-native (no
# container).
#
# fabtests is a 2-process server/client test (NOT mpirun-launched): we start the server
# on the first node bound to its EFA-reachable IP, then the client on the second node
# connects to that IP. Slurm places one task per node via srun.
#
# Submit (needs 2 nodes available):
#   sbatch -p gpu-p5en-spot fabtests-internode-efa.sbatch
#   BUF=cuda sbatch --export=ALL,BUF=cuda -p gpu-p5en-spot fabtests-internode-efa.sbatch

set -euo pipefail

###########################
###### User Variables #####
###########################

: "${BUILD_ROOT:=/fsx/p2p-bandwidth}"
: "${FORCE_REBUILD:=0}"
: "${FABTESTS_VERSION:=2.4.0}"                   # match installed libfabric (2.4.0amzn1)
: "${BUF:=host}"                                  # "host" or "cuda" (device buffers)
: "${ITERS:=100}"                                 # iterations per size (-I)
# fabtests sweeps a default size ladder when given "-S all"; that's the robust path
# (one server/client invocation) vs. restarting the server per size.

EFA_PREFIX="${EFA_PREFIX:-/opt/amazon/efa}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PKG_CONFIG_PATH="${EFA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${EFA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1

FT_PREFIX="${BUILD_ROOT}/fabtests-install-${FABTESTS_VERSION}"
FT_SRC="fabtests-${FABTESTS_VERSION}"
# fabtests is released as part of libfabric on GitHub.
FT_TGZ_URL="https://github.com/ofiwg/libfabric/releases/download/v${FABTESTS_VERSION}/fabtests-${FABTESTS_VERSION}.tar.bz2"

###########################
####### Preflight #########
###########################

echo "==================== Environment ===================="
echo "Date       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes      : ${SLURM_JOB_NUM_NODES:-?} (${SLURM_JOB_NODELIST:-?})"
echo "Slurm job  : ${SLURM_JOB_ID:-NA} / ${SLURM_JOB_PARTITION:-NA}"
echo "Buffers    : ${BUF}   (host = CPU<->CPU, cuda = GPU<->GPU GPUDirect RDMA)"
echo "libfabric  : $(${EFA_PREFIX}/bin/fi_info --version 2>/dev/null | head -1)"
srun -l hostname || true

###########################
######### Build ###########
###########################

FT_BW="${FT_PREFIX}/bin/fi_rdm_tagged_bw"
if [[ "${FORCE_REBUILD}" == "1" || ! -x "${FT_BW}" ]]; then
    echo
    echo "==================== Build fabtests ${FABTESTS_VERSION} (CUDA) ===================="
    mkdir -p "${BUILD_ROOT}"
    BUILD_TMP="${BUILD_ROOT}/build-${FT_SRC}"; rm -rf "${BUILD_TMP}"; mkdir -p "${BUILD_TMP}"
    pushd "${BUILD_TMP}" >/dev/null
    echo "Downloading ${FT_TGZ_URL} ..."
    curl -fSL -o fabtests.tar.bz2 "${FT_TGZ_URL}"
    tar xf fabtests.tar.bz2
    cd "fabtests-${FABTESTS_VERSION}"
    # Link against the installed EFA libfabric; enable CUDA for -D cuda device buffers.
    ./configure --with-libfabric="${EFA_PREFIX}" \
                --with-cuda="${CUDA_HOME}" \
                --prefix="${FT_PREFIX}"
    make -j"$(nproc)"
    make install
    popd >/dev/null
fi
[[ -x "${FT_BW}" ]] || { echo "ERROR: fi_rdm_tagged_bw not found after build"; exit 1; }
echo "fi_rdm_tagged_bw : ${FT_BW}"

###########################
########## Run ############
###########################

# Resolve the two node hostnames; node 0 = server, node 1 = client.
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
SERVER="${NODES[0]}"
CLIENT="${NODES[1]:-}"
[[ -n "${CLIENT}" ]] || { echo "ERROR: need 2 nodes, got: ${NODES[*]}"; exit 1; }
echo "Server node: ${SERVER}   Client node: ${CLIENT}"

# Device-buffer flag for fabtests: -D cuda selects CUDA memory for the test buffers.
DEV_ARGS=()
[[ "${BUF}" == "cuda" ]] && DEV_ARGS=(-D cuda)

echo
echo "==================== fabtests fi_rdm_tagged_bw over EFA (BUF=${BUF}) ===================="

# One invocation, full size ladder (-S all). Start the server on node 0 (binds and
# waits for the client), give it a moment, then run the client on node 1 pointed at the
# server hostname. The CLIENT prints the per-size bandwidth table that we parse/plot.
status=0
srun --nodes=1 --ntasks=1 --nodelist="${SERVER}" \
     "${FT_BW}" -p efa "${DEV_ARGS[@]}" -I "${ITERS}" -S all \
     > "/tmp/ft_srv_${SLURM_JOB_ID}.log" 2>&1 &
srv_pid=$!
sleep 5
srun --nodes=1 --ntasks=1 --nodelist="${CLIENT}" \
     "${FT_BW}" -p efa "${DEV_ARGS[@]}" -I "${ITERS}" -S all "${SERVER}" \
     || status=$?
wait "${srv_pid}" 2>/dev/null || true

echo
if [[ "${status}" -eq 0 ]]; then
    echo "RESULT: PASS — fabtests fi_rdm_tagged_bw (BUF=${BUF}) over EFA completed."
else
    echo "RESULT: FAIL — a fabtests run errored (exit ${status})."
    echo "Check FI_PROVIDER=efa, that both nodes can reach each other on EFA, and for"
    echo "BUF=cuda that GPUDirect RDMA (nvidia-peermem) is available."
fi

echo
echo "==================== How to read this ===================="
cat <<'EOF'
* fi_rdm_tagged_bw prints, per message size: bytes, iters, total, time, MB/sec, usec/xfer.
  The MB/sec column is the raw libfabric/EFA bandwidth for that size.
* This is the rawest inter-node number: no MPI, no NCCL — just libfabric on EFA. Compare:
    - osu-internode-efa.sbatch  (MPI over EFA)      — adds MPI/UCX overhead
    - nccl-internode-efa.sbatch (NCCL over EFA)     — adds the NCCL + OFI-plugin path
  All three ride the same EFA hardware, so fabtests is the practical ceiling for them.
* BUF=cuda uses GPUDirect RDMA (GPU memory over EFA); BUF=host is CPU memory over EFA.
* A single fi_rdm_tagged_bw pair drives one EFA path; full node bandwidth is multi-rail.
EOF
exit "${status}"
