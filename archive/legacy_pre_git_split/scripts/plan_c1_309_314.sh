#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
residual="${1:?Usage: bash scripts/plan_c1_309_314.sh <residual.pkl> <empty-output-dir>}"
out_dir="${2:?Usage: bash scripts/plan_c1_309_314.sh <residual.pkl> <empty-output-dir>}"
shift 2
c1_python="${C1_PYTHON:-$PWD/.venv-c1/bin/python}"
"$c1_python" -m single_integrator.c1.evaluate \
  --residual "$residual" --out-dir "$out_dir" \
  --initial-states baseline_309_314/planning/wide_initial_states_200.npz --split test --seed 42 \
  --n-rollouts 25 --max-steps 1200 "$@"
