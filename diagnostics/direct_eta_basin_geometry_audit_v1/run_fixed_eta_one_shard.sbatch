#!/bin/bash
#SBATCH --job-name=eta-fixed-arms
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1/logs/fixed-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1/logs/fixed-%j.err
set -euo pipefail
if [ "$#" -ne 2 ]; then echo 'usage: sbatch ... PLAN STAGE' >&2; exit 2; fi
D=/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
mkdir -p "$D/logs"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1
CUDA_VISIBLE_DEVICES=0 "$PY" "$D/run_fixed_eta_arms.py" --plan "$1" --stage "$2" --device gpu --batch 32
