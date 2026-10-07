#!/bin/bash
#SBATCH --job-name=of3-gap-init
#SBATCH --partition=debug
#SBATCH --array=0-1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_zero_active_gap_connectivity_v1/logs/init_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_zero_active_gap_connectivity_v1/logs/init_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.14
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/orthoflow3_zero_active_gap_connectivity_v1/run_gap_arms.py --plan diagnostics/orthoflow3_zero_active_gap_connectivity_v1/initial_screen_plan.json --stage initial_screen --device gpu --batch 32 --shard "$SLURM_ARRAY_TASK_ID" --shards 2
