# Intra-node GPU Peer-to-Peer Bandwidth

Two complementary single-node benchmarks for the **intra-node GPU↔GPU peer-to-peer**
path (NVLink / NVSwitch) on a GPU compute node — e.g. a `p5en.48xlarge` (8× H200) queue:

| Script | Tool | What it measures |
|---|---|---|
| [`p2p-bandwidth.sbatch`](./p2p-bandwidth.sbatch) | NVIDIA [`p2pBandwidthLatencyTest`](https://github.com/NVIDIA/cuda-samples) | Raw `cudaMemcpyPeer` bandwidth + latency matrices for **every** GPU pair, P2P enabled vs disabled |
| [`osu-p2p-mpi.sbatch`](./osu-p2p-mpi.sbatch) | **OpenMPI 5** + [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) (`osu_bw`, `osu_bibw`) | **CUDA-aware** MPI point-to-point bandwidth over NVLink, with device buffers (`D D`) |
| [`osu-p2p-mpi-hostbuf.sbatch`](./osu-p2p-mpi-hostbuf.sbatch) | OpenMPI 5 + OSU (`osu_bw`, `osu_bibw`) | **Non-CUDA-aware** baseline: the same OSU tests with host buffers (`H H`), CPU↔CPU only (no NVLink) |

Run all three to compare the hardware ceiling (NVIDIA tool), what CUDA-aware MPI
delivers on those links (OSU `D D`), and the non-CUDA-aware fallback (OSU `H H`) — see
[Comparing the results](#comparing-the-results). [`plot_results.py`](./plot_results.py)
turns the `.out` files into bandwidth-vs-message-size charts.

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
sbatch -p gpu-p5en-spot p2p-bandwidth.sbatch            # NVIDIA cudaMemcpyPeer matrices
sbatch -p gpu-p5en-spot osu-p2p-mpi.sbatch              # CUDA-aware MPI (device buffers)
sbatch -p gpu-p5en-spot osu-p2p-mpi-hostbuf.sbatch      # non-CUDA-aware baseline (host buffers)

# Wider message-size sweep (1 B .. 256 MiB) for the two OSU runs:
sbatch --export=ALL,MAX_BYTES=268435456 -p gpu-p5en-spot osu-p2p-mpi.sbatch
sbatch --export=ALL,MAX_BYTES=268435456 -p gpu-p5en-spot osu-p2p-mpi-hostbuf.sbatch
```

Output lands next to where you submit, as `<job-name>_<jobid>.out` (plus matching
`.err`). The first run builds the tools (~1–3 min); subsequent runs reuse the cache
under `/fsx/p2p-bandwidth` (all three scripts share one OSU build).

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
CUDA awareness is ~**37×** over the host path here. If the OSU device-buffer run *fails*
but the NVIDIA test looks healthy, the issue is MPI CUDA-awareness/config, not the GPUs
or NVLink. All report GB/s base-10, comparable to within a few percent.
