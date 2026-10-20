#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r18_freeze_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000M
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_eta2_gap_v1/freeze_test_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_eta2_gap_v1/freeze_test.py
