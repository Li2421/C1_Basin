#!/bin/bash
#SBATCH --job-name=of3_de_promote
#SBATCH --partition=debug
#SBATCH --array=0-5
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_direct_eta_baseline_v1/logs/promote_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_direct_eta_baseline_v1/logs/promote_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.14
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/orthoflow3_direct_eta_baseline_v1/run_target_rollouts.py --plan active_promotion_round1_plan.json --shard "${SLURM_ARRAY_TASK_ID}" --shards 6 --batch 32
