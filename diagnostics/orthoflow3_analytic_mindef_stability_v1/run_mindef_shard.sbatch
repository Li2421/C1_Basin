#!/bin/bash
#SBATCH --job-name=of3-mindef
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_analytic_mindef_stability_v1/logs/of3-mindef-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_analytic_mindef_stability_v1/logs/of3-mindef-%j.err
set -euo pipefail
BASE=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_analytic_mindef_stability_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1
CUDA_VISIBLE_DEVICES=0 "$PY" "$BASE/run_mindef.py" --plan "$1" --stage "$2" --shard "$3" --shards "$4" --batch 32
