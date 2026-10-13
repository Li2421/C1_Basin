#!/bin/bash
#SBATCH --job-name=c1-q-physical-fit
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/physical_state_critic_formal_v1/logs/fit_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
panels=(within_toy_TT within_toy_FF within_ring_TT within_db_FF within_four_FF loso_heldout_toy loso_heldout_db loso_heldout_four loso_heldout_ring)
seeds=(17 23 41)
panel_index=$((SLURM_ARRAY_TASK_ID / 6))
remainder=$((SLURM_ARRAY_TASK_ID % 6))
if (( remainder < 3 )); then kind=state; else kind=eta_only; fi
seed=${seeds[$((remainder % 3))]}
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.physical_state_critic_formal_v1.train_unified fit --panel "${panels[$panel_index]}" --kind "$kind" --seed "$seed"
