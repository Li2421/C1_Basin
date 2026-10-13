#!/bin/bash
#SBATCH --job-name=c1-q-cache-strict-fit
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/physical_state_critic_minimal_v1/logs/strict_fit_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export C1_Q_PILOT_VARIANT=strict
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
panels=(within_toy within_db within_four within_ring loso_toy loso_db loso_four loso_ring)
panel_index=$((SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then kind=state; else kind=eta_only; fi
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.physical_state_critic_minimal_v1.pilot fit --panel "${panels[$panel_index]}" --kind "$kind"
