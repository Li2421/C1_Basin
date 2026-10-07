#!/bin/bash
#SBATCH --job-name=c1-t0-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_true_t0_v1/post_%j.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_true_t0_v1/post_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu
DEST=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_true_t0_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -m diagnostics.orthoflow3_controller_information_probe_v1.held_true_t0_benchmark merge --stage "${BENCH_STAGE:-1}"
"$PY" -m shared_rollout_db.plan --manifest "$DEST/planned_stage${BENCH_STAGE:-1}.json" --output "$DEST/cache_postflight_stage${BENCH_STAGE:-1}.json" > /dev/null
"$PY" -m diagnostics.orthoflow3_controller_information_probe_v1.held_true_t0_evaluate --stage "${BENCH_STAGE:-1}"
