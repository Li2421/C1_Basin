#!/bin/bash
#SBATCH --job-name=c1-optimizer-audit
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/02_C1_Toy_GiveWay/results/c1_optimizer_diagnostic_v1-%j.log
set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for optimizer in adam sgd; do
  .venv-c1/bin/python -m single_integrator.c1.train_optimizer_diagnostic \
    --out "results/c1_optimizer_diagnostic_v1/${optimizer}" \
    --probe results/c1_guard_factorial_v1 --optimizer "$optimizer" \
    --warm-start results/c1_deadlock_union/baseline0_seed0/best_feasible.pkl
done
.venv-c1/bin/python scripts/analyze_c1_optimizer_diagnostic.py results/c1_optimizer_diagnostic_v1
