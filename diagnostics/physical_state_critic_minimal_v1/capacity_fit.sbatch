#!/bin/bash
#SBATCH --job-name=c1-q-capacity-fit
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/physical_state_critic_minimal_v1/logs/capacity_fit_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
panels=(within_toy within_db within_four within_ring loso_toy loso_db loso_four loso_ring)
if (( SLURM_ARRAY_TASK_ID < 8 )); then export C1_Q_PILOT_VARIANT=strict; else export C1_Q_PILOT_VARIANT=enriched; fi
panel_index=$((SLURM_ARRAY_TASK_ID % 8))
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.physical_state_critic_minimal_v1.pilot fit --panel "${panels[$panel_index]}" --kind eta_only_wide
