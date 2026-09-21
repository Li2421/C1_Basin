#!/bin/bash
#SBATCH --job-name=c1-risk-two-tests
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/c1_risk_two_tests_v1-%j.log
set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-c1/bin/python scripts/probe_c1_risk_two_tests.py --out results/c1_risk_two_tests_v1
.venv-c1/bin/python scripts/analyze_c1_risk_two_tests.py results/c1_risk_two_tests_v1
