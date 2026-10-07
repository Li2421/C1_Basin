#!/bin/bash
#SBATCH --job-name=k16-critic-v2
#SBATCH --partition=cpu
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%A_%a_train.out
#SBATCH --error=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%A_%a_train.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
training_seeds=(17 23 41)
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/orthoflow3_k16_critic_canonicalization_v1/train_phase_a.py --seed "${training_seeds[$SLURM_ARRAY_TASK_ID]}"
