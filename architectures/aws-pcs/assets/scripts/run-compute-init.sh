#!/usr/bin/env bash
# run-compute-init.sh — per-node compute-init runner.
#
# Fetched from S3 and invoked by the compute-node UserData (add-cng*.yaml) at first
# boot. Kept as a separate S3 script (not inline UserData) because PCS caps launch
# template UserData at 13384 bytes and add-cng.yaml is already close to that ceiling.
#
# Runs every regular file in the given directory ONCE, in stable (LC_ALL=C) sort
# order, as root, log-and-continue (a failing script is logged with its exit code
# but the node still joins the cluster — no drain/replace loop). No-op when the
# directory is empty/'none'/absent, so it is safe by default even without the S3 DRA.
#
# Usage:  run-compute-init.sh <scripts-dir>
#   env:  PCS_SLURM_VERSION exported to each script (matches the post-install hook).
#
# The default <scripts-dir> is /fsx/s3/compute-init — a folder under the FSx-Lustre
# S3 Data Repository Association mount — so scripts uploaded to the linked S3 bucket
# auto-appear on every compute node and run at launch.
set -u

CI_DIR="${1:-}"
CI_LOG=/var/log/pcs-compute-init.log

# Normalize for the 'none' sentinel (case-insensitive).
CI_LC="$(printf '%s' "${CI_DIR}" | tr '[:upper:]' '[:lower:]')"

if [ -z "${CI_DIR}" ] || [ "${CI_LC}" = "none" ]; then
  echo "compute-init: disabled (dir='${CI_DIR}')" | tee "${CI_LOG}"
  exit 0
fi

if [ ! -d "${CI_DIR}" ]; then
  echo "compute-init: nothing to do (dir '${CI_DIR}' absent — no S3 DRA, or nothing uploaded yet)" | tee "${CI_LOG}"
  exit 0
fi

echo "compute-init: running scripts in ${CI_DIR}" | tee "${CI_LOG}"

# Regular files only, at the top level of the folder, in stable C-locale order
# (prefix names 10-, 20-, … to control sequence). Subdirectories are ignored.
found=0
while IFS= read -r ci_script; do
  [ -z "${ci_script}" ] && continue
  found=1
  echo "compute-init: executing ${ci_script}" >> "${CI_LOG}"
  chmod +x "${ci_script}" 2>/dev/null || true
  bash "${ci_script}" >> "${CI_LOG}" 2>&1
  echo "compute-init: ${ci_script} exited $?" >> "${CI_LOG}"
done <<EOF
$(find "${CI_DIR}" -maxdepth 1 -type f | LC_ALL=C sort)
EOF

[ "${found}" -eq 0 ] && echo "compute-init: no regular files in ${CI_DIR}" >> "${CI_LOG}"
echo "compute-init: done" >> "${CI_LOG}"
exit 0
