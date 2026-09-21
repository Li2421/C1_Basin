#!/bin/bash
#SBATCH --job-name=c1-vi-vjp-one-sided
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=results/c1_vi_vjp_one_sided_v2-%j.log
set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-c1/bin/python scripts/diagnose_c1_vi_vjp_one_sided.py \
  --out-dir results/c1_vi_vjp_one_sided_v2
