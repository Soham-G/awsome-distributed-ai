#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Parse the p2p-bandwidth benchmark outputs and plot bandwidth vs message size.

Consumes the .out files produced by the three benchmarks in this directory:

  * osu-p2p-mpi.sbatch          -> CUDA-aware MPI, DEVICE buffers (D D), per GPU pair
  * osu-p2p-mpi-hostbuf.sbatch  -> non-CUDA-aware MPI, HOST buffers (H H), baseline
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
# OSU's own banner tells us the buffer mode reliably: "MPI-CUDA" => device buffers.
_BANNER = re.compile(r"#\s*OSU\s+MPI(-CUDA)?\s+(Bi-Directional\s+)?Bandwidth", re.IGNORECASE)


def parse_osu(path):
    """Return a list of series dicts: {label, test, mode, pair, sizes[], gbps[]}."""
    series = []
    cur = None
    pending_pair = None
    pending_test = None
    with open(path, errors="replace") as fh:
        for line in fh:
            hdr = _HDR.search(line)
            if hdr and "---" in line:
                # New labeled block from our sbatch wrapper.
                pending_test = "osu_" + hdr.group(1).lower()
                m = _PAIR.search(line)
                pending_pair = f"{m.group(1)}-{m.group(2)}" if m else None
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
    mode = "CUDA-aware (D D)" if s["mode"] == "device" else "host baseline (H H)"
    kind = "bidir" if s["test"] == "osu_bibw" else "unidir"
    pair = f" GPU {s['pair']}" if s["pair"] else ""
    return f"{mode} {kind}{pair}"


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="benchmark .out files")
    ap.add_argument("--out-dir", default=".", help="directory for PNG output")
    ap.add_argument("--title", default="Intra-node GPU P2P bandwidth (p5en, 8x H200)")
    args = ap.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")

    osu_series = []
    nvidia_peaks = {}
    for path in args.files:
        if not os.path.exists(path):
            print(f"warning: {path} not found, skipping", file=sys.stderr)
            continue
        # Classify by content. Use a SPECIFIC marker that only the real NVIDIA matrix
        # output contains — the "P2P=Enabled ... Bandwidth ... Matrix" header line —
        # not the tool name "p2pBandwidthLatencyTest", which also appears in the OSU
        # scripts' prose footer and would misclassify an OSU run. An OSU run is
        # recognized by its "# OSU ... Bandwidth Test" banner. A file can satisfy
        # both (build logs etc.), so prefer whichever parser actually yields data.
        with open(path, errors="replace") as fh:
            body = fh.read()
        is_nvidia = re.search(r"P2P=Enabled.*Bandwidth.*Matrix", body) is not None
        osu_found = parse_osu(path)
        if osu_found:
            osu_series += osu_found
            print(f"{path}: parsed {len(osu_found)} OSU series")
        elif is_nvidia:
            nvidia_peaks = parse_nvidia_peaks(path) or nvidia_peaks
            print(f"{path}: NVIDIA peaks {nvidia_peaks}")
        else:
            print(f"{path}: no recognizable benchmark data")

    if not osu_series and not nvidia_peaks:
        sys.exit("no parseable benchmark data found in the given files")

    os.makedirs(args.out_dir, exist_ok=True)

    # One figure per direction (unidirectional / bidirectional) so the CUDA-aware vs
    # host baseline contrast is clean and the y-scale isn't dominated by one mode.
    for kind, test in (("Unidirectional", "osu_bw"), ("Bidirectional", "osu_bibw")):
        group = [s for s in osu_series if s["test"] == test]
        if not group:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for s in sorted(group, key=lambda x: (x["mode"] != "device", x["label"])):
            style = "-o" if s["mode"] == "device" else "--s"
            ax.plot(s["sizes"], s["gbps"], style, markersize=4, label=s["label"])
        peak = nvidia_peaks.get("unidir" if test == "osu_bw" else "bidir")
        if peak:
            ax.axhline(peak, color="black", ls=":", lw=1.2,
                       label=f"NVIDIA cudaMemcpyPeer peak ~{peak:.0f} GB/s")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Message size (bytes)")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title(f"{args.title}\n{kind} P2P bandwidth vs message size")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
        out = os.path.join(args.out_dir, f"p2p_bandwidth_{test}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
