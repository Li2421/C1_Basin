#!/bin/bash
#SBATCH --job-name=contract-dev
#SBATCH --partition=cpu
#SBATCH --array=0-17%18
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=diagnostics/orthoflow3_closed_loop_contract_v1/%A_%a_dev.out
#SBATCH --error=diagnostics/orthoflow3_closed_loop_contract_v1/%A_%a_dev.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_closed_loop_contract_v1.evaluate run --shard "$SLURM_ARRAY_TASK_ID" --shards 18
