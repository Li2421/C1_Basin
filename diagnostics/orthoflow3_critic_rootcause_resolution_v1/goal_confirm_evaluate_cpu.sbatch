#!/bin/bash
#SBATCH --job-name=c1-goal-target-evalcpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_eval_cpu_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_eval_cpu_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.goal_evaluation --replicate "$SLURM_ARRAY_TASK_ID"
