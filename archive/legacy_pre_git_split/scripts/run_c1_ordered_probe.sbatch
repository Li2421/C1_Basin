#!/bin/bash
#SBATCH --job-name=c1-ordered-probe
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/02_C1_Toy_GiveWay/results/c1_gradient_ordered_v1.slurm-%j.log
set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if .venv-c1/bin/python -c 'import json,sys; d=json.load(open("results/c1_exact_margin_v1/complete.json")); sys.exit(0 if d["status"]=="no_validation_feasible_checkpoint" else 1)'; then
    .venv-c1/bin/python scripts/probe_c1_gradient_closed_loop.py --out results/c1_gradient_ordered_v1 --cases 40 --candidate ordered --replay-budget-seconds 1500
    .venv-c1/bin/python scripts/audit_c1_risk_ordering.py results/c1_gradient_ordered_v1 --candidate ordered
    .venv-c1/bin/python scripts/analyze_c1_gradient_closed_loop.py results/c1_gradient_ordered_v1
fi
