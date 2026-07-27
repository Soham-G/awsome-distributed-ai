#!/usr/bin/env bash
# Slurm Prolog: GPU Health Check
# Runs fast checks (0, 2) before job execution by default.
# Non-zero exit causes Slurm to drain the node and requeue the job.
#
# Default behavior (~8 seconds):
#   Runs check 0 (nvidia-smi) and check 2 (EFA enumeration) only.
#   Prolog output goes to syslog / slurmd logs, not job output files.
#
# Configuration in slurm.conf:
#   Prolog=/path/to/prolog-gpu-healthcheck.sh
#   PrologTimeout=900   # 15 minutes
#
# Environment variable overrides:
#   GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1  -- Enable DCGM L2 (check 1) in prolog

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEALTHCHECK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HEALTHCHECK_SCRIPT="${HEALTHCHECK_DIR}/gpu-healthcheck.sh"

# Log to syslog for Slurm prolog visibility
log_prolog() {
    logger -t "gpu-healthcheck-prolog" "$*"
    echo "[gpu-healthcheck-prolog] $*" >&2
}

# ─── GPU presence gate ───────────────────────────────────────────────────────
# Slurm's Prolog is cluster-wide (slurm.conf), and AWS PCS only supports Prolog at the
# cluster level (not per compute node group). So on a mixed cluster (e.g. a cpu1 queue +
# a GPU queue) this prolog also fires on CPU/login nodes, where the GPU checks would fail
# and wrongly drain the node. Detect GPU presence first and cleanly skip (exit 0) on
# non-GPU nodes — this is what makes a single cluster-wide Prolog safe and effectively
# scopes the health check to GPU node groups only.
node_has_gpu() {
    # Prefer a PCI scan (works even if the driver/nvidia-smi is broken — which is exactly
    # the failure we want to CATCH on a GPU node, not skip). Fall back to nvidia-smi.
    if command -v lspci >/dev/null 2>&1; then
        lspci -d 10de: 2>/dev/null | grep -qiE '3d controller|vga|nvidia' && return 0
    fi
    [[ -e /dev/nvidia0 ]] && return 0
    command -v nvidia-smi >/dev/null 2>&1 && return 0
    return 1
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    if ! node_has_gpu; then
        log_prolog "No GPU detected on $(hostname) — skipping GPU health check prolog (exit 0)"
        exit 0
    fi

    log_prolog "Starting GPU health check prolog for job ${SLURM_JOB_ID:-unknown}"
    log_prolog "Node: $(hostname), User: ${SLURM_JOB_USER:-unknown}"

    # Set results directory under /tmp with job context
    export RESULTS_DIR="/tmp/gpu-healthcheck-prolog-${SLURM_JOB_ID:-$$}"

    # DCGM L2 is off by default (adds minutes of silence before job output).
    # Set GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1 to opt in.
    if [[ "${GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM:-0}" != "1" ]]; then
        log_prolog "Running fast prolog (checks 0, 2 only; set GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1 to include DCGM L2)"
        # Run only checks 0 and 2
        local exit_code=0
        bash "${HEALTHCHECK_DIR}/checks/0-nvidia-smi-check.sh" || exit_code=$?
        if [[ ${exit_code} -ne 0 ]]; then
            log_prolog "FAIL: nvidia-smi check failed -- draining node"
            exit 1
        fi

        bash "${HEALTHCHECK_DIR}/checks/2-efa-enumeration.sh" || exit_code=$?
        if [[ ${exit_code} -ne 0 ]]; then
            log_prolog "FAIL: EFA enumeration failed -- draining node"
            exit 1
        fi

        log_prolog "PASS: Prolog checks completed (DCGM skipped)"
        exit 0
    fi

    # Run full prolog suite (checks 0-2) including DCGM L2
    log_prolog "Running full prolog suite including DCGM L2 (GPU_HEALTHCHECK_PROLOG_ENABLE_DCGM=1)"
    local exit_code=0
    bash "${HEALTHCHECK_SCRIPT}" --prolog --results-dir "${RESULTS_DIR}" || exit_code=$?

    if [[ ${exit_code} -ne 0 ]]; then
        log_prolog "FAIL: Prolog health checks failed (exit ${exit_code}) -- draining node"
        log_prolog "Results: ${RESULTS_DIR}"
        exit 1
    fi

    log_prolog "PASS: All prolog health checks passed"

    # Clean up results on success (optional -- comment out to retain)
    rm -rf "${RESULTS_DIR}" 2>/dev/null || true

    exit 0
}

main "$@"
