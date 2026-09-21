#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cache_dirs=$(find . -type d -name '__pycache__' -printf . | wc -c)
cache_files=$(find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -printf . | wc -c)
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
echo "Removed ${cache_dirs} Python cache directories and ${cache_files} bytecode files."

if [[ "${1:-}" == "--outputs" ]]; then
  rollouts=$(find single_integrator/results -type f \( -name 'rollout_*.npz' -o -name 'rollout_*.png' \) -printf . | wc -c)
  snapshots=$(find single_integrator -type d \( -name source_snapshot -o -name final_source_snapshot \) -printf . | wc -c)
  legacy_evals=$(find flowbc -type d \( -name 'eval_*' -o -name eval_sweep \) -printf . | wc -c)
  find single_integrator/results -type f \( -name 'rollout_*.npz' -o -name 'rollout_*.png' \) -delete
  find single_integrator -type d \( -name source_snapshot -o -name final_source_snapshot \) -prune -exec rm -rf {} +
  find flowbc -type d \( -name 'eval_*' -o -name eval_sweep \) -prune -exec rm -rf {} +
  echo "Removed ${rollouts} rollout artifacts, ${snapshots} source snapshots, and ${legacy_evals} legacy evaluation directories."
elif [[ $# -gt 0 ]]; then
  echo "Unknown option: $1" >&2
  exit 2
fi
