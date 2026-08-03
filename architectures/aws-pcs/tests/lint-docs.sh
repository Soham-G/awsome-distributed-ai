#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Docs ⇄ template consistency lint for the AWS PCS reference architecture.
#
# Catches the most common drift after a parameter rename/removal: docs (and
# other templates) still referencing an old parameter name, a removed parameter
# presented as current, or a known-stale phrase. Run from anywhere; paths are
# resolved relative to this script's location (architectures/aws-pcs/).
#
#   bash architectures/aws-pcs/tests/lint-docs.sh
#
# Exit code is non-zero if any check fails, so it can gate a PR in CI.
#
# When you intentionally rename/remove a parameter, update the BANNED list below
# in the same change — that is the point: the lint forces docs to keep up.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # architectures/aws-pcs
cd "$ROOT"

# Files that describe the user-facing interface (must not mention removed params).
DOC_GLOBS=(README.md docs/*.md tests/*.md)

fail=0
report() { echo "FAIL: $1"; fail=1; }

# 1. Removed / renamed parameters must not appear in docs as if current.
#    Each entry is TAB-separated: <banned-regex>\t<allowed-context-regex>\t<message>.
#    A hit is a failure UNLESS the line also matches the allowed-context regex
#    (used to permit explicit "(renamed from X)" / "→" / "internally" history
#    notes). Use NEVERMATCH as the allow-regex when no exception applies.
BANNED=(
  $'OnDemandEnableEfa\tremoved\tOnDemandEnableEfa was replaced by OnDemandEfaInterfaceCount (0/1/2)'
  $'GrafanaPublicAccessCidr\t[Rr]enamed|→ ?`?GrafanaAccessCidr\tGrafanaPublicAccessCidr was renamed to GrafanaAccessCidr'
  $'DeployMonitoring=true\tMonitoringStack|internally|nested\tDeployMonitoring (bool) was replaced by MonitoringStack at the deploy-all layer'
  $'S3 (public )?hosting is not allowed\tNEVERMATCH\tPostInstallScriptUrl now accepts s3:// URLs'
  $'architectures/aws-pcs/iam/\tNEVERMATCH\tthe iam/ directory was removed; use docs/IAM.md + assets/cluster-*-iam.yaml'
)
for entry in "${BANNED[@]}"; do
  IFS=$'\t' read -r pat allow msg <<<"$entry"
  hits=$(grep -rnE "$pat" "${DOC_GLOBS[@]}" 2>/dev/null | grep -vE "$allow" || true)
  if [ -n "$hits" ]; then
    report "stale reference ($msg):"
    echo "$hits" | sed 's/^/    /'
  fi
done

# 2. PostInstallScriptUrl skip sentinel. Two wrong patterns to catch:
#    (a) empty string shown as skip — empty now AUTO-INSTALLS from the templates bucket;
#    (b) a whitespace-only value ("single space") shown as skip — CloudFormation TRIMS a
#        whitespace-only param to empty, so it never skips; it takes the default-installer
#        branch. The correct skip sentinel is the literal `none`.
hits=$(grep -rnE 'PostInstallScriptUrl=""' "${DOC_GLOBS[@]}" 2>/dev/null || true)
if [ -n "$hits" ]; then
  report 'PostInstallScriptUrl="" (empty) shown as skip — empty auto-installs; use the literal `none` to skip:'
  echo "$hits" | sed 's/^/    /'
fi
# Flag a whitespace-only ParameterValue (CFN trims it → never skips). Only the literal
# value form is a hard error; explanatory prose that says "a single space does NOT skip"
# is allowed (it's the correction we want to keep).
hits=$(grep -rnE "ParameterValue=(\" \"|' ')" "${DOC_GLOBS[@]}" 2>/dev/null || true)
if [ -n "$hits" ]; then
  report "whitespace-only ParameterValue shown — CloudFormation trims it to empty (never a skip). Use the literal 'none':"
  echo "$hits" | sed 's/^/    /'
fi
# Flag prose that PRESCRIBES a single space to skip (but allow the negated correction).
hits=$(grep -rniE "single space" "${DOC_GLOBS[@]}" 2>/dev/null \
       | grep -ivE "not[^.]*skip|does \*\*not\*\*|never skip|do not use a single space|does not skip" || true)
if [ -n "$hits" ]; then
  report "docs prescribe a single space to skip — CFN trims whitespace-only to empty (runs the default installer). Use the literal 'none' to skip:"
  echo "$hits" | sed 's/^/    /'
fi

# 3. Every parameter the deploy-all template declares should be documented in
#    PARAMETERS.md (catches a new param added without a docs row).
params=$(awk '/^Parameters:/{p=1;next} /^[A-Za-z]/{p=0} p&&/^  [A-Z][A-Za-z0-9]+:/{gsub(/[: ]/,"");print}' assets/pcs-ml-cluster-deploy-all.yaml | sort -u)
for prm in $params; do
  grep -q "\`$prm\`" docs/PARAMETERS.md || report "deploy-all parameter '$prm' is not documented in docs/PARAMETERS.md"
done

# 4. Same-file Markdown anchor links in README.md resolve to a real heading.
#    (Cross-file and external links are out of scope — kept simple on purpose.)
while IFS= read -r anchor; do
  # build the set of heading slugs in README
  slugs=$(grep -E '^#{1,6} ' README.md \
    | sed -E 's/^#{1,6} //; s/`//g' \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9 -]//g; s/ /-/g')
  grep -qx "$anchor" <<<"$slugs" || report "README.md internal anchor '#$anchor' has no matching heading"
done < <(grep -oE '\]\(#[a-z0-9-]+\)' README.md | sed -E 's/\]\(#//; s/\)//' | sort -u)

if [ "$fail" -eq 0 ]; then
  echo "docs lint: PASS (no stale parameter references, no empty=skip wording, all deploy-all params documented, README anchors resolve)"
fi
exit $fail
