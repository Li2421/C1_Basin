#!/bin/bash
#SBATCH --job-name=r26_fit
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=5000M
#SBATCH --time=06:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/method_benchmark_v1/r26_fit_%A_%a.log
#SBATCH --gres=shard:1
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export JAX_PLATFORMS=cuda
for k in {0..6}; do
 for seed in 17 23 41; do
  "$PY" -B diagnostics/method_benchmark_v1/train_m3.py cv "$k" "$seed"
 done
done
exec "$PY" -B diagnostics/method_benchmark_v1/train_m3.py full
