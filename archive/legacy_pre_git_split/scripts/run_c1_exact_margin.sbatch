#!/bin/bash
#SBATCH --job-name=c1-exact-full
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/02_C1_Toy_GiveWay/results/c1_exact_margin_v1.slurm-%j.log

set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv-c1/bin/python scripts/audit_c1_risk_ordering.py results/c1_gradient_exact_v1
.venv-c1/bin/python -m single_integrator.c1.train_exact_margin \
    --out results/c1_exact_margin_v1 --probe results/c1_gradient_exact_v1 --updates 64
