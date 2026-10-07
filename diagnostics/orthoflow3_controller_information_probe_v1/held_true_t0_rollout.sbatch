#!/bin/bash
#SBATCH --job-name=c1-t0-b15
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_true_t0_v1/rollout_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_true_t0_v1/rollout_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
test -f diagnostics/orthoflow3_controller_information_probe_v1/held_controller_true_t0_v1/frozen_predictions.json
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_information_probe_v1.held_true_t0_benchmark \
  run --stage "${BENCH_STAGE:-1}" --shard "$SLURM_ARRAY_TASK_ID" --shards 6
