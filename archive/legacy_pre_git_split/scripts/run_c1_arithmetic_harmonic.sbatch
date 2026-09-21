#!/bin/bash
#SBATCH --job-name=c1-ah-risk-probe
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=results/c1_arithmetic_harmonic_probe_v1-%j.log
set -euo pipefail
cd /home/zhihan/research/02_C1_Toy_GiveWay
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-c1/bin/python scripts/probe_c1_arithmetic_harmonic.py --out results/c1_arithmetic_harmonic_probe_v1
.venv-c1/bin/python scripts/analyze_c1_arithmetic_harmonic.py results/c1_arithmetic_harmonic_probe_v1
