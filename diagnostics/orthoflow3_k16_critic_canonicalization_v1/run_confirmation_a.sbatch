#!/bin/bash
#SBATCH --job-name=k16-A-confirm
#SBATCH --partition=cpu
#SBATCH --array=0-35%18
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=7-00:00:00
#SBATCH --output=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%A_%a_confirm.out
#SBATCH --error=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%A_%a_confirm.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/orthoflow3_k16_critic_canonicalization_v1/confirmation_a.py run --shard "${SLURM_ARRAY_TASK_ID}" --shards 36
