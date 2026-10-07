#!/bin/bash
#SBATCH --job-name=ortho3-hygiene-replay
#SBATCH --partition=cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/orthoflow3_data_hygiene_v2/logs/replay_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_data_hygiene_v2/logs/replay_%A_%a.err

set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
scenarios=(double_bottleneck four_way_intersection ring_exchange)
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_data_hygiene_v2/replay.py run \
  --scenario "${scenarios[$SLURM_ARRAY_TASK_ID]}"
