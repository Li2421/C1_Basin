#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r17_a3all
#SBATCH --cpus-per-task=4
#SBATCH --mem=10000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_critic_new_recipe_confirmation_v1/fit_a3_all_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_critic_new_recipe_confirmation_v1/fit_a3_all.py
