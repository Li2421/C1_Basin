#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
dataset="${1:-datasets/give_way_si_short_v1}"
shift $(( $# > 0 ? 1 : 0 ))
python -m single_integrator.validate "$dataset" "$@"
