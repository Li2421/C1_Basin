#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=r14a_newflows
#SBATCH --array=0-13%3
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/train_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_independent_flow_confirmation_v2/train_planned.py "$SLURM_ARRAY_TASK_ID"
