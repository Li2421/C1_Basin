#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
out_dir="${1:?Usage: bash scripts/generate_dataset.sh <empty-output-dir> [generate options]}"
shift
python -m single_integrator.generate --out_dir "$out_dir" "$@"
