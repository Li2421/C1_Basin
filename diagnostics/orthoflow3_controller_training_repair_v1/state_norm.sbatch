#!/bin/bash
#SBATCH --job-name=c1-repair-state-scale
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_training_repair_v1/state_norm_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_training_repair_v1/state_norm_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
seeds=(17 23 41)
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_controller_training_repair_v1.train train --kind physical_context_state_normalized --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"
