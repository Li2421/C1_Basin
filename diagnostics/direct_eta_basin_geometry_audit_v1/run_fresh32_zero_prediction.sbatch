#!/bin/bash
#SBATCH --job-name=eta-fresh32-zp
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1/logs/fresh32-zp-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1/logs/fresh32-zp-%j.err
set -euo pipefail
D=/home/zhihan/research/Basin_C1/diagnostics/direct_eta_basin_geometry_audit_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
mkdir -p "$D/logs"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1
CUDA_VISIBLE_DEVICES=0 "$PY" "$D/run_fresh32_zero_prediction.py" --device gpu --batch 32
