#!/bin/bash
#SBATCH --job-name=c1-state-info
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:40:00
#SBATCH --output=diagnostics/orthoflow3_state_context_adjudication_v1/micro_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_state_context_adjudication_v1/micro_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
controllers=(base alt second)
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_state_context_adjudication_v1.micro_audit --controller "${controllers[$SLURM_ARRAY_TASK_ID]}"
