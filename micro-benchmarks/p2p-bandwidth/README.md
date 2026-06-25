# Intra-node GPU Peer-to-Peer Bandwidth

Two complementary single-node benchmarks for the **intra-node GPU↔GPU peer-to-peer**
path (NVLink / NVSwitch) on a GPU compute node — e.g. a `p5en.48xlarge` (8× H200) queue:

| Script | Tool | What it measures |
|---|---|---|
| [`p2p-bandwidth.sbatch`](./p2p-bandwidth.sbatch) | NVIDIA [`p2pBandwidthLatencyTest`](https://github.com/NVIDIA/cuda-samples) | Raw `cudaMemcpyPeer` bandwidth + latency matrices for **every** GPU pair, P2P enabled vs disabled |
| [`osu-p2p-mpi.sbatch`](./osu-p2p-mpi.sbatch) | **OpenMPI 5** + [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) (`osu_bw`, `osu_bibw`) | CUDA-aware **MPI** point-to-point bandwidth over the same NVLink path, with device buffers (`D D`) |

Run both to compare the hardware ceiling (NVIDIA tool) against what CUDA-aware MPI
actually delivers on those links (OSU) — see [Comparing the two](#comparing-the-two).

> **Scope:** this is *intra-node* P2P (GPU↔GPU inside one node over NVLink). For
> *multi-node* GPU communication over EFA, use the
> [NCCL tests](../nccl-tests/) instead.

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
sbatch -p gpu-p5en-spot p2p-bandwidth.sbatch     # Benchmark 1 (NVIDIA)
sbatch -p gpu-p5en-spot osu-p2p-mpi.sbatch       # Benchmark 2 (OpenMPI 5 + OSU)
```

Output lands next to where you submit, as `p2p-bandwidth_<jobid>.out` and
`osu-p2p-mpi_<jobid>.out` (plus matching `.err`). The first run builds the tools
(~1–3 min); subsequent runs reuse the cache under `/fsx/p2p-bandwidth`.

### Environment knobs

| Variable | Default | Applies to | Purpose |
|---|---|---|---|
| `BUILD_ROOT` | `/fsx/p2p-bandwidth` | both | Where clones/builds/binaries are cached |
| `FORCE_REBUILD` | `0` | both | `1` = rebuild even if a cached binary exists |
| `CUDA_SAMPLES_REF` | `master` | NVIDIA | cuda-samples git ref to build |
| `OSU_VERSION` | `7.5` | OSU | OSU Micro-Benchmarks release to build |
| `GPU_PAIRS` | `0:1 0:7` | OSU | Space-separated `a:b` GPU pairs to test |
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

**Benchmark 2** prints `osu_bw` (unidirectional, peak GB/s at large messages) and
`osu_bibw` (bidirectional) for each pair in `GPU_PAIRS`, using device send/recv buffers.
The `RESULT: PASS` line confirms CUDA-aware device transfers worked on OpenMPI 5.

### Comparing the two

For a given GPU pair `a↔b`:

| OSU (MPI) | ≈ NVIDIA matrix cell |
|---|---|
| `osu_bw`   (unidirectional) | Unidirectional Bandwidth Matrix (P2P Enabled) `[a][b]` |
| `osu_bibw` (bidirectional)  | Bidirectional  Bandwidth Matrix (P2P Enabled) `[a][b]` |

CUDA-aware MPI typically lands **somewhat below** the raw `cudaMemcpyPeer` ceiling
because of UCX/MPI protocol and staging overhead — that gap is expected. Both should be
far above PCIe. If the OSU device-buffer run *fails* but Benchmark 1 looks healthy, the
issue is MPI CUDA-awareness/config, not the GPUs or NVLink. Both tools report GB/s in
base-10, so the numbers are directly comparable to within a few percent.
