# GPU Peer-to-Peer Bandwidth (intra-node NVLink & inter-node EFA)

Benchmarks for GPU↔GPU peer-to-peer bandwidth, both **intra-node** (over NVLink /
NVSwitch, on one GPU node) and **inter-node** (over EFA, between two nodes) — e.g. on a
`p5en.48xlarge` (8× H200) queue. Run several tools over the same links to see the
hardware ceiling, what CUDA-aware MPI delivers, the non-CUDA-aware fallback, and what
NCCL (what real training uses) achieves.

**Intra-node (NVLink):**

| Script | Tool | What it measures |
|---|---|---|
| [`p2p-bandwidth.sbatch`](./p2p-bandwidth.sbatch) | NVIDIA [`p2pBandwidthLatencyTest`](https://github.com/NVIDIA/cuda-samples) | Raw `cudaMemcpyPeer` bandwidth + latency matrices for **every** GPU pair (single 160 MB transfer). The hardware ceiling. |
| [`osu-p2p-mpi.sbatch`](./osu-p2p-mpi.sbatch) | **OpenMPI 5** + [OSU](https://mvapich.cse.ohio-state.edu/benchmarks/) (`osu_bw`, `osu_bibw`) | **CUDA-aware** MPI point-to-point over NVLink, device buffers (`D D`) |
| [`osu-p2p-mpi-hostbuf.sbatch`](./osu-p2p-mpi-hostbuf.sbatch) | OpenMPI 5 + OSU | **Non-CUDA-aware** baseline: same tests, host buffers (`H H`), CPU↔CPU only |
| [`nccl-sendrecv.sbatch`](./nccl-sendrecv.sbatch) | [nccl-tests](https://github.com/NVIDIA/nccl-tests) `sendrecv_perf` (Pyxis container) | **NCCL** point-to-point over NVLink, message-size sweep (busbw) |

**Inter-node (EFA):**

| Script | Tool | What it measures |
|---|---|---|
| [`osu-internode-efa.sbatch`](./osu-internode-efa.sbatch) | OpenMPI 5 + OSU, 2 nodes | MPI point-to-point **over EFA** between 2 nodes; `D D` uses GPUDirect RDMA, `H H` is host↔host |
| [`nccl-internode-efa.sbatch`](./nccl-internode-efa.sbatch) | nccl-tests `sendrecv_perf`, 2 nodes (Pyxis) | **NCCL over EFA** between 2 nodes — the path real multi-node training uses |

[`plot_results.py`](./plot_results.py) turns all the `.out` files into
bandwidth-vs-message-size charts, including an intra-vs-inter (NVLink vs EFA) overlay —
see [Plotting](#plotting) and [Comparing the results](#comparing-the-results).

> **Related:** for full multi-node collective benchmarks (all-reduce at scale over EFA),
> see the [NCCL tests](../nccl-tests/). The inter-node scripts here are deliberately
> minimal (single GPU pair) for an apples-to-apples P2P comparison with the intra-node
> numbers.

## Prerequisites

- A GPU queue on the cluster (e.g. `gpu-p5en-spot`). Submit with `sbatch -p <queue>`.
- The **PCS-Ready DLAMI** toolchain on the node (already present): `nvcc`, NVIDIA driver,
  and **OpenMPI 5** at `/opt/amazon/openmpi5`.
- A shared filesystem at **`/fsx`** — used to cache the compiled binaries so re-runs
  (and other nodes) skip the build.

> **OSU + CUDA 13:** the OSU benchmark builds against a **CUDA 12** toolkit on purpose.
> CUDA 13 changed the `cudaMemPrefetchAsync()` signature and OSU ≤ 7.5 won't compile
> against it. `osu-p2p-mpi.sbatch` auto-selects the newest `/usr/local/cuda-12.*` on the
> DLAMI (the H200 driver runs CUDA 12 binaries fine); override with `CUDA_HOME=...`. The
> NVIDIA tool builds fine on CUDA 13.

No container is used — both scripts build and run DLAMI-native.

## Usage

```bash
# From the login node (these scale the queue up from 0 on first submit):
sbatch -p gpu-p5en-spot p2p-bandwidth.sbatch            # NVIDIA cudaMemcpyPeer matrices
sbatch -p gpu-p5en-spot osu-p2p-mpi.sbatch              # CUDA-aware MPI (device buffers)
sbatch -p gpu-p5en-spot osu-p2p-mpi-hostbuf.sbatch      # non-CUDA-aware baseline (host buffers)

# Wider message-size sweep (1 B .. 256 MiB) for the two OSU runs:
sbatch --export=ALL,MAX_BYTES=268435456 -p gpu-p5en-spot osu-p2p-mpi.sbatch
sbatch --export=ALL,MAX_BYTES=268435456 -p gpu-p5en-spot osu-p2p-mpi-hostbuf.sbatch
```

Output lands next to where you submit, as `<job-name>_<jobid>.out` (plus matching
`.err`). The first run builds the tools (~1–3 min); subsequent runs reuse the cache
under `/fsx/p2p-bandwidth` (all OSU scripts share one OSU build; both NCCL scripts share
one cached `.sqsh`).

### Running the full matrix

[`run-all.sh`](./run-all.sh) submits every benchmark in order (intra-node first, then the
2-node EFA jobs) with the 256 MiB sweep, from the login node:

```bash
cd /fsx && bash run-all.sh gpu-p5en-spot     # arg = partition; defaults to gpu-p5en-spot
```

The inter-node jobs need **2 nodes available at once** — on a Spot queue that depends on
capacity (they queue until two `p5en` instances launch). After everything completes, plot:

```bash
python3 plot_results.py --out-dir plots *_*.out
```

### Plotting

[`plot_results.py`](./plot_results.py) (matplotlib) parses the `.out` files and writes
bandwidth-vs-message-size PNGs (log-x), one per direction, overlaying the CUDA-aware
`D D` curves, the host `H H` baseline, and the NVIDIA `cudaMemcpyPeer` peak as a
reference line:

```bash
python3 plot_results.py --out-dir plots \
    osu-p2p-mpi_*.out osu-p2p-mpi-hostbuf_*.out p2p-bandwidth_*.out
```

It needs only matplotlib and the `.out` files — run it on the login node (the PCS-Ready
DLAMI ships matplotlib) or anywhere you've copied the outputs.

### Environment knobs

| Variable | Default | Applies to | Purpose |
|---|---|---|---|
| `BUILD_ROOT` | `/fsx/p2p-bandwidth` | all | Where clones/builds/binaries are cached |
| `FORCE_REBUILD` | `0` | all | `1` = rebuild even if a cached binary exists |
| `CUDA_SAMPLES_REF` | `master` | NVIDIA | cuda-samples git ref to build |
| `OSU_VERSION` | `7.5` | OSU | OSU Micro-Benchmarks release to build |
| `MAX_BYTES` | *(unset)* | OSU | Max message size in bytes for the sweep (e.g. `268435456` = 256 MiB). Unset = OSU default (≤ 4 MiB) |
| `GPU_PAIRS` | `0:1 0:7` | OSU `D D` | Space-separated `a:b` GPU pairs to test (device run only) |
| `OMPI` | `/opt/amazon/openmpi5` | OSU | OpenMPI install to use (kept on **5.x** on purpose) |

Example — test more pairs at higher rebuild:
```bash
GPU_PAIRS="0:1 0:2 0:7 3:4" sbatch -p gpu-p5en-spot osu-p2p-mpi.sbatch
```

The GPU architecture is **auto-detected** (`nvidia-smi --query-gpu=compute_cap`), so both
scripts also work unchanged on the `gpu-g7e-*` and P6/B200 queues.

## Reading the results

**Benchmark 1** prints six matrices; the ones that matter are the **P2P Enabled**
Unidirectional and Bidirectional Bandwidth matrices. Every off-diagonal cell is
GPU *i* → GPU *j*. The job first prints `nvidia-smi topo -m` — off-diagonal entries
should be `NV#` (NVLink); `PIX`/`PXB`/`SYS` would mean a pair is only reachable over
PCIe/host.

On a healthy **p5en (8× H200, NVSwitch all-to-all)** expect bidirectional P2P-enabled
bandwidth roughly **700–900 GB/s** for every pair, and **P2P Enabled ≫ P2P Disabled**.
A pair stuck near PCIe speeds (tens of GB/s) while `topo -m` showed `NV#` points to a
degraded link — cross-check `nvidia-smi nvlink -e`.

**The OSU runs** print `osu_bw` (unidirectional, peak GB/s at large messages) and
`osu_bibw` (bidirectional). The CUDA-aware run (`osu-p2p-mpi.sbatch`) does this per pair
in `GPU_PAIRS` with device buffers; its `RESULT: PASS` line confirms CUDA-aware device
transfers worked on OpenMPI 5. The host-buffer run (`osu-p2p-mpi-hostbuf.sbatch`) is the
non-CUDA-aware CPU↔CPU baseline (no GPU pair loop — it never touches the GPUs).

### Comparing the results

For a given GPU pair `a↔b`:

| Measurement | Compare against |
|---|---|
| OSU `D D` `osu_bw` (CUDA-aware, unidir) | NVIDIA Unidirectional Bandwidth Matrix (P2P Enabled) `[a][b]` |
| OSU `D D` `osu_bibw` (CUDA-aware, bidir) | NVIDIA Bidirectional Bandwidth Matrix (P2P Enabled) `[a][b]` |
| OSU `H H` (host baseline) | divide `D D` by `H H` → the CUDA-aware speedup |

CUDA-aware MPI (`D D`) typically lands **somewhat below** the raw `cudaMemcpyPeer`
ceiling (NVIDIA tool) because of UCX/MPI protocol and staging overhead — that gap is
expected — but **far above** the host (`H H`) baseline, which is bounded by CPU/shared-
memory bandwidth and never uses NVLink. Measured on a p5en (8× H200), large-message
unidirectional: NVIDIA ~393 GB/s, CUDA-aware MPI ~311 GB/s, host baseline ~8 GB/s — so
CUDA awareness is ~**37×** over the host path here. **NCCL** (`nccl-sendrecv.sbatch`)
typically sits between the raw peer-copy and CUDA-aware MPI. If the OSU device-buffer run
*fails* but the NVIDIA test looks healthy, the issue is MPI CUDA-awareness/config, not the
GPUs or NVLink. All report GB/s base-10, comparable to within a few percent.

### Intra-node (NVLink) vs inter-node (EFA)

The inter-node scripts measure the **same point-to-point pattern across the network over
EFA** instead of within the node over NVLink. Expect inter-node bandwidth to plateau
**well below** intra-node — a single GPU-pair over one EFA path is an order of magnitude
under NVLink/NVSwitch — which is exactly why training keeps as much traffic intra-node as
possible. Things to look for:

- **OSU `D D` over EFA** uses **GPUDirect RDMA** (NIC DMAs GPU memory directly); the
  `H H` variant is plain host↔host EFA — the gap between them is the GPUDirect benefit.
- **NCCL over EFA** must show `NET/OFI Selected provider is efa ... (found N nics)` in
  the `NCCL_DEBUG` log; if it falls back to TCP sockets, bandwidth drops sharply.
- The plotter's `p2p_bandwidth_overlay.png` puts NVLink and EFA curves on one **log-y**
  chart so the ~10× gap is visible at a glance (only emitted when both scopes are present).
- A single pair drives one "rail"; full inter-node bandwidth needs all 8 GPUs
  (multi-rail) — see the [NCCL tests](../nccl-tests/) for the at-scale all-reduce.
