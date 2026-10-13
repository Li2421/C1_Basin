#!/bin/bash
#SBATCH --job-name=c1-ring-qstate
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/physical_state_critic_formal_v1/logs/ring_train_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
seeds=(17 23 41)
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
if ((SLURM_ARRAY_TASK_ID < 3)); then kind=state; else kind=eta_only; fi
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.physical_state_critic_formal_v1.train_ring_ff "$kind" --seed "$seed"
