#!/bin/bash
#SBATCH --job-name=of3-migration
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_representation_migration_v1/logs/of3-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_representation_migration_v1/logs/of3-%j.err
set -euo pipefail
if [ "$#" -ne 4 ]; then echo 'usage: PLAN STAGE SHARD SHARDS' >&2; exit 2; fi
D=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_representation_migration_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1
CUDA_VISIBLE_DEVICES=0 "$PY" "$D/run_subset_arms.py" --plan "$1" --stage "$2" --shard "$3" --shards "$4" --device gpu --batch 32
