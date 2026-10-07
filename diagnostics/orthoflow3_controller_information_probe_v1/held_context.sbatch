#!/bin/bash
#SBATCH --job-name=c1-held-H20
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_robust_selection_v1/context_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_robust_selection_v1/context_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_information_probe_v1.held_robust_predict \
  features --shard "$SLURM_ARRAY_TASK_ID" --shards 4
