#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Parse the p2p-bandwidth benchmark outputs and plot bandwidth vs message size.

Consumes the .out files produced by the benchmarks in this directory:

  * osu-p2p-mpi.sbatch          -> CUDA-aware MPI, DEVICE buffers (D D), per GPU pair
  * osu-p2p-mpi-hostbuf.sbatch  -> non-CUDA-aware MPI, HOST buffers (H H), baseline
  * nccl-sendrecv.sbatch        -> NCCL sendrecv_perf sweep (busbw), 2 GPUs
  * p2p-bandwidth.sbatch        -> NVIDIA p2pBandwidthLatencyTest (matrix, optional)

Each OSU section is a block like:

    -------------------- osu_bw  (...): GPU 0 <-> GPU 1 ... --------------------
    # OSU MPI-CUDA Bandwidth Test v7.5
    # Size      Bandwidth (MB/s)
    1            0.23
    ...

We parse every such block into a labeled series (test, buffer-mode, pair) and plot
bandwidth (GB/s) vs message size (bytes) on a log-x axis. The NVIDIA matrices are a
single fixed-size point, so they're optionally drawn as horizontal reference lines
(their peak per-pair P2P-enabled bandwidth) rather than curves.

Usage:
    python3 plot_results.py [--out-dir DIR] FILE [FILE ...]

    # typical: point it at the sweep outputs pulled back from the cluster
    python3 plot_results.py --out-dir plots \\
        osu-p2p-mpi_*.out osu-p2p-mpi-hostbuf_*.out p2p-bandwidth_*.out

Dependencies: matplotlib (pip install matplotlib). No GPU / cluster access needed —
run it anywhere you've copied the .out files.
"""

import argparse
import os
import re
import sys

# OSU table row: "<size>   <bandwidth MB/s>" (osu_bw / osu_bibw report 2 columns).
_ROW = re.compile(r"^\s*(\d+)\s+([0-9]+\.[0-9]+)\s*$")
# OSU section header this repo's sbatch scripts emit, e.g.
#   "------ osu_bw  (unidirectional, ...): GPU 0 <-> GPU 1 ... ------"
#   "------ osu_bibw (bidirectional, host buffers) [H H] ------"
_HDR = re.compile(r"osu_(bw|bibw)\b(.*)", re.IGNORECASE)
_PAIR = re.compile(r"GPU\s+(\d+)\s*<->\s*GPU\s+(\d+)")
# Transport tag the sbatch scripts put in the section header, e.g. "[ob1-ofi]" / "[ucx]".
# (cm-ofi is the default; older outputs without a tag are treated as cm-ofi.)
_XPORT = re.compile(r"\[(cm-ofi|ob1-ofi|ucx)\]")
# OSU's own banner tells us the buffer mode reliably: "MPI-CUDA" => device buffers.
_BANNER = re.compile(r"#\s*OSU\s+MPI(-CUDA)?\s+(Bi-Directional\s+)?Bandwidth", re.IGNORECASE)


def parse_osu(path):
    """Return a list of series dicts: {label, test, mode, pair, sizes[], gbps[]}."""
    series = []
    cur = None
    pending_pair = None
    pending_test = None
    pending_xport = None
    with open(path, errors="replace") as fh:
        for line in fh:
            hdr = _HDR.search(line)
            if hdr and "---" in line:
                # New labeled block from our sbatch wrapper.
                pending_test = "osu_" + hdr.group(1).lower()
                m = _PAIR.search(line)
                pending_pair = f"{m.group(1)}-{m.group(2)}" if m else None
                x = _XPORT.search(line)
                pending_xport = x.group(1) if x else None
                cur = None
                continue
            ban = _BANNER.search(line)
            if ban:
                # Start collecting rows; banner disambiguates device vs host.
                mode = "device" if ban.group(1) else "host"
                test = pending_test or ("osu_bibw" if ban.group(2) else "osu_bw")
                cur = {
                    "test": test,
                    "mode": mode,
                    "pair": pending_pair,
                    "xport": pending_xport,
                    "sizes": [],
                    "gbps": [],
                }
                cur["label"] = _mk_label(cur)
                series.append(cur)
                continue
            row = _ROW.match(line)
            if row and cur is not None:
                size = int(row.group(1))
                mbps = float(row.group(2))
                cur["sizes"].append(size)
                cur["gbps"].append(mbps / 1000.0)  # MB/s -> GB/s (base-10, matches OSU)
    # Drop empty blocks.
    return [s for s in series if s["sizes"]]


def _mk_label(s):
    if s["mode"] == "device":
        # Name the MPI transport so cm-ofi / ob1-ofi / ucx are distinct series.
        xport = s.get("xport") or "cm-ofi"
        mode = f"MPI/{xport} (D D)"
    else:
        mode = "host baseline (H H)"
    kind = "bidir" if s["test"] == "osu_bibw" else "unidir"
    pair = f" GPU {s['pair']}" if s["pair"] else ""
    return f"{mode} {kind}{pair}"


# nccl-tests data row: "size count type redop root time algbw busbw error ..." for
# out-of-place plus the same for in-place. Columns vary, but the first field is the
# byte size and busbw is the 2nd-to-last numeric on each of the two halves. We key off
# the leading integer size and pull the standard nccl-tests layout:
#   size(B) count(elem) type redop root  time algbw busbw #wrong  time algbw busbw #wrong
_NCCL_ROW = re.compile(r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+\S+\s+"
                       r"[\d.]+\s+[\d.]+\s+([\d.]+)\s+\S+")


def parse_nccl(path):
    """Parse nccl-tests sendrecv_perf output -> one series of (size, busbw GB/s).

    nccl-tests already reports busbw in GB/s (base-2-ish, but directly comparable to the
    other GB/s numbers within a few %). Returns [] if the file isn't an nccl-tests run."""
    sizes, gbps = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _NCCL_ROW.match(line)
            if m:
                sizes.append(int(m.group(1)))
                gbps.append(float(m.group(2)))  # out-of-place busbw, already GB/s
    if not sizes:
        return []
    return [{"test": "nccl", "mode": "nccl", "pair": None, "sizes": sizes,
             "gbps": gbps, "label": "NCCL sendrecv (busbw)"}]


# fabtests fi_rdm_tagged_bw table row:
#   bytes  iters  total      time   MB/sec   usec/xfer  Mxfers/sec
# e.g. "65536   100   6.40m  0.00s   4521.13   14.50      0.07"
# The size is the leading integer; MB/sec is the 5th numeric column. total/time carry
# unit suffixes (k/m/g, s), so match them loosely and pick the bandwidth field.
_FAB_ROW = re.compile(
    r"^\s*(\d+)\s+\d+\s+[\d.]+[kmgKMG]?\s+[\d.]+s?\s+([\d.]+)\s+[\d.]+\s+[\d.]+")


def parse_fabtests(path):
    """Parse fabtests fi_rdm_tagged_bw output -> one series of (size, GB/s).

    fabtests reports MB/sec (base-10 MB); convert to GB/s for comparison. The buffer
    mode (host vs cuda) is taken from the script's 'Buffers : ...' preflight line."""
    mode = "host"
    sizes, gbps = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "Buffers" in line and "cuda" in line.lower():
                mode = "cuda"
            m = _FAB_ROW.match(line)
            if m:
                sizes.append(int(m.group(1)))
                gbps.append(float(m.group(2)) / 1000.0)  # MB/s -> GB/s
    if not sizes:
        return []
    tag = "device/GPUDirect" if mode == "cuda" else "host"
    return [{"test": "fabtests", "mode": "fabtests", "pair": None, "sizes": sizes,
             "gbps": gbps, "label": f"libfabric fabtests ({tag})"}]


def parse_nvidia_peaks(path):
    """Extract peak per-pair bandwidth from the NVIDIA p2pBandwidthLatencyTest matrices.

    Returns {'unidir': gbps, 'bidir': gbps} using the max off-diagonal cell of the
    P2P=Enabled matrices (a single fixed-size measurement -> reference line)."""
    peaks = {}
    want = None
    with open(path, errors="replace") as fh:
        for line in fh:
            low = line.lower()
            if "unidirectional" in low and "p2p=enabled" in low:
                want = "unidir"; vals = []; continue
            if "bidirectional" in low and "p2p=enabled" in low:
                want = "bidir"; vals = []; continue
            if want:
                nums = re.findall(r"\d+\.\d+", line)
                if nums:
                    # Off-diagonal cells are the smaller values; diagonal is the
                    # multi-TB/s self-copy. Keep cells < 2000 GB/s (exclude self).
                    vals += [float(n) for n in nums if float(n) < 2000.0]
                elif vals:
                    peaks[want] = max(vals)
                    want = None
    return peaks


def _scope_of(path):
    """intra-node (NVLink) vs inter-node (EFA), inferred from the filename."""
    base = os.path.basename(path).lower()
    return "inter" if "internode" in base else "intra"


def _no_placement_group(body):
    """True if the inter-node run's annotation shows nodes had no cluster placement
    group (the Spot default). The scripts print 'placement_group=(none)' per node."""
    return "placement_group=(none)" in body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="benchmark .out files")
    ap.add_argument("--out-dir", default=".", help="directory for PNG output")
    ap.add_argument("--title", default="GPU P2P bandwidth (p5en, 8x H200)")
    args = ap.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    osu_series = []
    nccl_series = []
    fabtests_series = []
    nvidia_peaks = {}
    inter_no_pg = False   # any inter-node run launched without a placement group?
    for path in args.files:
        if not os.path.exists(path):
            print(f"warning: {path} not found, skipping", file=sys.stderr)
            continue
        # Classify by content. Use a SPECIFIC marker that only the real NVIDIA matrix
        # output contains — the "P2P=Enabled ... Bandwidth ... Matrix" header line —
        # not the tool name "p2pBandwidthLatencyTest", which also appears in the OSU
        # scripts' prose footer and would misclassify an OSU run. OSU runs are
        # recognized by their "# OSU ... Bandwidth Test" banner, NCCL by its data rows.
        # A file can satisfy several markers (build logs etc.), so prefer whichever
        # parser actually yields data.
        with open(path, errors="replace") as fh:
            body = fh.read()
        is_nvidia = re.search(r"P2P=Enabled.*Bandwidth.*Matrix", body) is not None
        scope = _scope_of(path)
        if scope == "inter" and _no_placement_group(body):
            inter_no_pg = True
        osu_found = parse_osu(path)
        nccl_found = parse_nccl(path) if "busbw" in body else []
        fab_found = parse_fabtests(path) if "MB/sec" in body else []
        for s in osu_found + nccl_found + fab_found:
            s["scope"] = scope
            net = "NVLink" if scope == "intra" else "EFA"
            s["label"] = f"{s['label']} [{net}]"
        if osu_found:
            osu_series += osu_found
            print(f"{path}: parsed {len(osu_found)} OSU series ({scope}-node)")
        elif nccl_found:
            nccl_series += nccl_found
            print(f"{path}: parsed NCCL series ({len(nccl_found[0]['sizes'])} sizes, {scope}-node)")
        elif fab_found:
            fabtests_series += fab_found
            print(f"{path}: parsed fabtests series ({len(fab_found[0]['sizes'])} sizes, {scope}-node)")
        elif is_nvidia:
            nvidia_peaks = parse_nvidia_peaks(path) or nvidia_peaks
            print(f"{path}: NVIDIA peaks {nvidia_peaks}")
        else:
            print(f"{path}: no recognizable benchmark data")

    if not (osu_series or nccl_series or fabtests_series or nvidia_peaks):
        sys.exit("no parseable benchmark data found in the given files")

    os.makedirs(args.out_dir, exist_ok=True)

    # Caveat stamped on any chart that includes inter-node EFA data taken without a
    # cluster placement group (the Spot default) — latency/bandwidth can be more variable.
    PG_NOTE = ("Note: inter-node (EFA) nodes ran WITHOUT a cluster placement group "
               "(Spot default) —\ninter-node latency/bandwidth may be higher/more variable "
               "than a placement-grouped pair.")

    def _stamp(fig, series_on_fig):
        if inter_no_pg and any(s.get("scope") == "inter" for s in series_on_fig):
            fig.text(0.5, 0.005, PG_NOTE, ha="center", va="bottom", fontsize=7,
                     color="dimgray", wrap=True)

    # Style every series uniquely so the legend is unambiguous:
    #   * marker encodes the TOOL  (OSU device = o, OSU host = s, NCCL = ^, fabtests = D)
    #   * linestyle encodes the SCOPE (solid = intra/NVLink, dashed = inter/EFA)
    #   * color is assigned explicitly per series (matplotlib's tab10 cycle), so two
    #     series can never collapse to the same color+marker+linestyle in the legend.
    _CYCLE = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])

    def _marker(s):
        if s["test"] == "nccl":     return "^"
        if s["test"] == "fabtests": return "D"
        if s["mode"] != "device":   return "s"          # OSU host baseline
        # OSU device: distinct marker per MPI transport.
        return {"cm-ofi": "o", "ob1-ofi": "v", "ucx": "P"}.get(s.get("xport") or "cm-ofi", "o")

    def _linestyle(s):
        return "-" if s.get("scope", "intra") == "intra" else "--"

    def _plot_series(ax, series):
        # Stable order (intra before inter, then by label) and a distinct color each.
        for i, s in enumerate(sorted(series, key=lambda x: (x.get("scope") != "intra", x["label"]))):
            ax.plot(s["sizes"], s["gbps"], marker=_marker(s), linestyle=_linestyle(s),
                    color=_CYCLE[i % len(_CYCLE)], markersize=4, label=s["label"])

    # One figure per direction (unidirectional / bidirectional) so the CUDA-aware vs
    # host baseline contrast is clean and the y-scale isn't dominated by one mode.
    # NCCL sendrecv (a point-to-point send/recv) is plotted on the unidirectional figure
    # alongside the OSU osu_bw curves.
    for kind, test in (("Unidirectional", "osu_bw"), ("Bidirectional", "osu_bibw")):
        group = [s for s in osu_series if s["test"] == test]
        extra = nccl_series if test == "osu_bw" else []
        if not group and not extra:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        _plot_series(ax, group + extra)
        peak = nvidia_peaks.get("unidir" if test == "osu_bw" else "bidir")
        if peak:
            # The NVIDIA tool transfers a single 160 MB buffer (numElems=40M ints), so
            # this is a large-message reference point, not a sweep — label it as such.
            ax.axhline(peak, color="black", ls=":", lw=1.2,
                       label=f"NVIDIA cudaMemcpyPeer @160MB ~{peak:.0f} GB/s")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Message size (bytes)")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title(f"{args.title}\n{kind} P2P bandwidth vs message size")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
        out = os.path.join(args.out_dir, f"p2p_bandwidth_{test}.png")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _stamp(fig, group + extra)
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")

    # Overlay: every unidirectional-style curve (OSU osu_bw + NCCL sendrecv + libfabric
    # fabtests), both scopes, on one log-log chart so intra-node NVLink vs inter-node EFA
    # is directly comparable. Only emit it when more than one scope is present.
    overlay = ([s for s in osu_series if s["test"] == "osu_bw"]
               + list(nccl_series) + list(fabtests_series))
    scopes = {s.get("scope", "intra") for s in overlay}
    if overlay and len(scopes) > 1:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        _plot_series(ax, overlay)   # same unique tool-marker / scope-linestyle scheme
        for k, lbl in (("unidir", "NVIDIA cudaMemcpyPeer @160MB (NVLink)"),):
            if nvidia_peaks.get(k):
                ax.axhline(nvidia_peaks[k], color="black", ls=":", lw=1.2,
                           label=f"{lbl} ~{nvidia_peaks[k]:.0f} GB/s")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")           # log-y: NVLink and EFA differ by ~10x
        ax.set_xlabel("Message size (bytes)")
        ax.set_ylabel("Bandwidth (GB/s, log)")
        ax.set_title(f"{args.title}\nIntra-node (NVLink) vs inter-node (EFA) — "
                     "unidirectional P2P")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
        out = os.path.join(args.out_dir, "p2p_bandwidth_overlay.png")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _stamp(fig, overlay)
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
