#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
checkpoint="${1:?Usage: bash scripts/evaluate_c1_geometry.sh <checkpoint> <empty-output-dir> [val|test] [evaluate options]}"
out_dir="${2:?Usage: bash scripts/evaluate_c1_geometry.sh <checkpoint> <empty-output-dir> [val|test] [evaluate options]}"
split="${3:-test}"
if (( $# >= 3 )); then shift 3; else shift $#; fi
python -m single_integrator.evaluate --checkpoint "$checkpoint" --out_dir "$out_dir" --split "$split" \
  --action_filter single_integrator.cbf:cbf_factory --c1_deadlock_geometry "$@"
