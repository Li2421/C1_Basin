#!/bin/bash
#SBATCH --job-name=c1-grad-probe
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/zhihan/research/02_C1_Toy_GiveWay/results/c1_gradient_closed_loop_v1.slurm-%j.log

set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-c1/bin/python \
  scripts/probe_c1_gradient_closed_loop.py \
  --out results/c1_gradient_closed_loop_v1 \
  --cases 40 \
  --replay-budget-seconds 1800
