#!/bin/bash
#SBATCH --job-name=giveway-residual-effect
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/giveway_residual_effect_pilot_v1/slurm-%j.log
set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-c1/bin/python scripts/pilot_giveway_residual_effect.py \
  --out results/giveway_residual_effect_pilot_v1
