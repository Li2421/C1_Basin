#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
checkpoint="${1:?Usage: bash scripts/evaluate_cbf.sh <checkpoint> <empty-output-dir> [val|test] [CBF options]}"
out_dir="${2:?Usage: bash scripts/evaluate_cbf.sh <checkpoint> <empty-output-dir> [val|test] [CBF options]}"
split="${3:-val}"
if (( $# >= 3 )); then shift 3; else shift $#; fi
python -m single_integrator.evaluate_cbf --checkpoint "$checkpoint" --out_dir "$out_dir" --split "$split" "$@"
