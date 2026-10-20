#!/bin/bash
#SBATCH --job-name=r26_blind_gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=7000M
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/method_benchmark_v1/r26_blind_gate_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/method_benchmark_v1/predict.py
exec "$PY" -B diagnostics/method_benchmark_v1/reconstruction_gate.py
