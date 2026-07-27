#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Submit the full GPU P2P bandwidth matrix (intra-node NVLink + inter-node EFA) to a PCS
# Slurm queue, with the 256 MiB message-size sweep. Run from the login node:
#
#   bash run-all.sh [partition]      # default partition: gpu-p5en-spot
#
# Then, once all jobs complete, render the charts:
#   python3 plot_results.py --out-dir plots *_*.out
#
# Notes:
#   * Intra-node jobs need 1 GPU node; inter-node jobs need 2 nodes AT ONCE (on a Spot
#     queue they'll stay PENDING until two instances launch).
#   * All scripts cache builds under /fsx/p2p-bandwidth, so only the first of each tool
#     pays the build/import cost.

set -euo pipefail

PARTITION="${1:-gpu-p5en-spot}"
SWEEP=268435456   # 256 MiB
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting GPU P2P bandwidth matrix to partition: ${PARTITION}"
echo "(message-size sweep up to ${SWEEP} bytes / 256 MiB)"
echo

sub () {  # sub <description> <sbatch args...>
    local desc="$1"; shift
    printf '  %-44s ' "${desc}"
    sbatch -p "${PARTITION}" "$@"
}

echo "== Intra-node (NVLink) =="
sub "NVIDIA p2pBandwidthLatencyTest"      "${HERE}/p2p-bandwidth.sbatch"
sub "OSU CUDA-aware (D D)"                --export=ALL,MAX_BYTES=${SWEEP} "${HERE}/osu-p2p-mpi.sbatch"
sub "OSU host baseline (H H)"             --export=ALL,MAX_BYTES=${SWEEP} "${HERE}/osu-p2p-mpi-hostbuf.sbatch"
sub "NCCL sendrecv"                       "${HERE}/nccl-sendrecv.sbatch"

echo
echo "== Inter-node (EFA, needs 2 nodes) =="
sub "libfabric fabtests, host"            --export=ALL,BUF=host "${HERE}/fabtests-internode-efa.sbatch"
sub "libfabric fabtests, device (cuda)"   --export=ALL,BUF=cuda "${HERE}/fabtests-internode-efa.sbatch"
sub "OSU over EFA, device (D D)"          --export=ALL,BUFFER_TYPE=D\ D "${HERE}/osu-internode-efa.sbatch"
sub "OSU over EFA, host (H H)"            --export=ALL,BUFFER_TYPE=H\ H "${HERE}/osu-internode-efa.sbatch"
sub "NCCL over EFA"                       "${HERE}/nccl-internode-efa.sbatch"

echo
echo "Submitted. Watch with:  squeue"
echo "When done, plot with:   python3 ${HERE}/plot_results.py --out-dir plots *_*.out"
