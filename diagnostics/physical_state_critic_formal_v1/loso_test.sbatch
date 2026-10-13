#!/bin/bash
#SBATCH --job-name=c1-q-loso-test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/physical_state_critic_formal_v1/logs/loso_test_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
index=$((SLURM_ARRAY_TASK_ID + ${ARRAY_OFFSET:-0}))
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.physical_state_critic_formal_v1.loso_worker run --mode test --index "$index"
