#!/bin/bash
#SBATCH --job-name=c1-ordered-full
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/02_C1_Toy_GiveWay/results/c1_ordered_parallel_v2.slurm-%j.log
set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv-c1/bin/python -m single_integrator.c1.train_ordered_parallel \
    --out results/c1_ordered_parallel_v2 --probe results/c1_gradient_ordered_v1 \
    --updates 64 --feasibility-stage --training-seed 2026091728 \
    --warm-start results/c1_deadlock_union/baseline0_seed0/best_feasible.pkl
