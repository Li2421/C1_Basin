#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=r18_critic
#SBATCH --array=0-3%2
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000M
#SBATCH --time=12:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_eta2_gap_v1/critic_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
models=(C1 C3 C2 A0)
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_eta2_gap_v1/critic_train.py "${models[$SLURM_ARRAY_TASK_ID]}"
