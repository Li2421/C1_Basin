#!/bin/bash
#SBATCH --job-name=structured_q
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_structured_continuous_q_data_v1/logs/shard_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_structured_continuous_q_data_v1/logs/shard_%A_%a.err
set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11 /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_structured_continuous_q_data_v1/run_shard.py --shard "${SLURM_ARRAY_TASK_ID}"
