#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
out_dir="${1:?Usage: bash scripts/train.sh <empty-output-dir> [steps] [train options]}"
shift
steps="${1:-100000}"
if (( $# > 0 )); then shift; fi
python -m single_integrator.train --dataset datasets/give_way_si_short_v1 --training_data_dir datasets/give_way_si_short_uniform_state_v1/raw --out_dir "$out_dir" --steps "$steps" --normalize "$@"
