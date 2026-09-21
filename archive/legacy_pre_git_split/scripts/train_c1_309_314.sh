#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out_dir="${1:?Usage: bash scripts/train_c1_309_314.sh <empty-output-dir> <seed-0-or-1> [C1 options]}"
model_seed="${2:?Specify 0 or 1}"
shift 2
case "$model_seed" in
  0|1) ;;
  *) echo "seed must be 0 or 1" >&2; exit 2 ;;
esac
c1_python="${C1_PYTHON:-$PWD/.venv-c1/bin/python}"
"$c1_python" -m single_integrator.c1.train \
  --checkpoint "baseline_309_314/checkpoints/seed${model_seed}/ckpt_0025000.pkl" \
  --out-dir "$out_dir" --seed "$model_seed" "$@"
