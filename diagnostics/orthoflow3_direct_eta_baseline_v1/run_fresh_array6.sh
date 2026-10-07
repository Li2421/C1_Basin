#!/bin/bash
#SBATCH --job-name=of3-de-fresh
#SBATCH --partition=debug
#SBATCH --array=0-5
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_direct_eta_baseline_v1/logs/fresh_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_direct_eta_baseline_v1/logs/fresh_%A_%a.err
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.14
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_direct_eta_baseline_v1/run_fresh_wide.py --shard "$SLURM_ARRAY_TASK_ID" --shards 6
