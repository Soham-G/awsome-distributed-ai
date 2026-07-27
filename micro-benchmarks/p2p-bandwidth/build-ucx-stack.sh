#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Build a UCX-enabled OpenMPI + a CUDA OSU linked against it, cached on /fsx, for the
# INTRA-node UCX comparison (UCX cuda_ipc transport over NVLink). This is a separate
# stack because the EFA-bundled /opt/amazon/openmpi5 is OFI-only (no UCX).
#
# IMPORTANT scope: UCX here is for INTRA-node GPU<->GPU only. EFA is not exposed as a
# verbs/RDMA device on these nodes (empty /sys/class/infiniband), so UCX cannot drive
# EFA for inter-node — it would fall back to TCP/ENA. Use this stack only on 1 node.
#
# No GPU needed to BUILD (just nvcc for OSU's CUDA kernels), so run it on the login node:
#   bash build-ucx-stack.sh
# It is idempotent: skips anything already built under BUILD_ROOT.

set -euo pipefail

: "${BUILD_ROOT:=/fsx/p2p-bandwidth}"
: "${UCX_VERSION:=1.18.0}"
: "${OMPI_UCX_VERSION:=5.0.6}"          # OpenMPI to build against UCX (any 5.x is fine)
: "${OSU_VERSION:=7.5}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"

UCX_PREFIX="${BUILD_ROOT}/ucx-install-${UCX_VERSION}"
OMPI_PREFIX="${BUILD_ROOT}/openmpi-ucx-${OMPI_UCX_VERSION}"
OSU_UCX_PREFIX="${BUILD_ROOT}/osu-install-${OSU_VERSION}-ucx"
SRC="${BUILD_ROOT}/src-ucx"
mkdir -p "${SRC}"

export PATH="${CUDA_HOME}/bin:${PATH}"
NJ="$(nproc)"

echo "==================== UCX stack build ===================="
echo "UCX ${UCX_VERSION} -> ${UCX_PREFIX}"
echo "OpenMPI ${OMPI_UCX_VERSION} (--with-ucx) -> ${OMPI_PREFIX}"
echo "OSU ${OSU_VERSION} (CUDA, against UCX-OMPI) -> ${OSU_UCX_PREFIX}"
echo "CUDA: ${CUDA_HOME}  cores: ${NJ}"

# ---- 1. UCX (CUDA-enabled; cuda + cuda_copy + cuda_ipc transports) ----
if [[ -x "${UCX_PREFIX}/bin/ucx_info" ]]; then
    echo "[1/3] UCX already built — skipping"
else
    echo "[1/3] Building UCX ${UCX_VERSION} ..."
    cd "${SRC}"
    [[ -f "ucx-${UCX_VERSION}.tar.gz" ]] || \
        curl -fSL -o "ucx-${UCX_VERSION}.tar.gz" \
        "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz"
    rm -rf "ucx-${UCX_VERSION}"; tar xf "ucx-${UCX_VERSION}.tar.gz"; cd "ucx-${UCX_VERSION}"
    # No verbs/rdmacm needed for intra-node cuda_ipc; enable CUDA, disable the IB bits.
    ./configure --prefix="${UCX_PREFIX}" \
        --with-cuda="${CUDA_HOME}" \
        --without-rdmacm --without-verbs \
        --enable-mt --enable-optimizations
    make -j"${NJ}"
    make install
fi

# ---- 2. OpenMPI --with-ucx ----
if [[ -x "${OMPI_PREFIX}/bin/mpicc" ]]; then
    echo "[2/3] OpenMPI(UCX) already built — skipping"
else
    echo "[2/3] Building OpenMPI ${OMPI_UCX_VERSION} --with-ucx ..."
    cd "${SRC}"
    series="v${OMPI_UCX_VERSION%.*}"
    [[ -f "openmpi-${OMPI_UCX_VERSION}.tar.bz2" ]] || \
        curl -fSL -o "openmpi-${OMPI_UCX_VERSION}.tar.bz2" \
        "https://download.open-mpi.org/release/open-mpi/${series}/openmpi-${OMPI_UCX_VERSION}.tar.bz2"
    rm -rf "openmpi-${OMPI_UCX_VERSION}"; tar xf "openmpi-${OMPI_UCX_VERSION}.tar.bz2"
    cd "openmpi-${OMPI_UCX_VERSION}"
    # Build with UCX + CUDA. Leave OFI out so PML/ucx is the clear default on this build.
    ./configure --prefix="${OMPI_PREFIX}" \
        --with-ucx="${UCX_PREFIX}" \
        --with-cuda="${CUDA_HOME}" --with-cuda-libdir="${CUDA_HOME}/lib64/stubs" \
        --without-ofi --disable-sphinx
    make -j"${NJ}"
    make install
fi

# ---- 3. OSU against the UCX OpenMPI ----
if [[ -x "$(find "${OSU_UCX_PREFIX}" -name osu_bw 2>/dev/null | head -1)" ]]; then
    echo "[3/3] OSU(UCX) already built — skipping"
else
    echo "[3/3] Building OSU ${OSU_VERSION} against UCX-OpenMPI ..."
    cd "${SRC}"
    [[ -f "osu-micro-benchmarks-${OSU_VERSION}.tar.gz" ]] || \
        curl -fSL -o "osu-micro-benchmarks-${OSU_VERSION}.tar.gz" \
        "https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz"
    rm -rf "osu-micro-benchmarks-${OSU_VERSION}"; tar xf "osu-micro-benchmarks-${OSU_VERSION}.tar.gz"
    cd "osu-micro-benchmarks-${OSU_VERSION}"
    ./configure CC="${OMPI_PREFIX}/bin/mpicc" CXX="${OMPI_PREFIX}/bin/mpicxx" \
        --enable-cuda --with-cuda="${CUDA_HOME}" --prefix="${OSU_UCX_PREFIX}"
    make -j"${NJ}"
    make install
fi

echo
echo "==================== UCX stack ready ===================="
echo "ucx_info : ${UCX_PREFIX}/bin/ucx_info"
echo "mpirun   : ${OMPI_PREFIX}/bin/mpirun"
echo "osu_bw   : $(find "${OSU_UCX_PREFIX}" -name osu_bw 2>/dev/null | head -1)"
"${UCX_PREFIX}/bin/ucx_info" -v 2>/dev/null | head -2 || true
"${OMPI_PREFIX}/bin/ompi_info" 2>/dev/null | grep -i ucx | head -3 || true
