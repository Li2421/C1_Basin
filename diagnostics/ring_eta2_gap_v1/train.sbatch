#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=r18_train
#SBATCH --array=0-35%2
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_eta2_gap_v1/train_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_eta2_gap_v1/train_one.py "$SLURM_ARRAY_TASK_ID"
