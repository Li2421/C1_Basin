#!/bin/bash
#SBATCH --job-name=c1-risk-predict
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/02_C1_Toy_GiveWay/results/c1_conditional_gpu_v2-%j.log
set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv-c1/bin/python scripts/probe_c1_conditional_risk_cpu.py --backend gpu --out results/c1_conditional_gpu_v2
.venv-c1/bin/python scripts/analyze_c1_conditional_risk.py results/c1_conditional_gpu_v2
