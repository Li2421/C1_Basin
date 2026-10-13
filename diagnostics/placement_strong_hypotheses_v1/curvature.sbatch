#!/bin/bash
#SBATCH --job-name=c1-strong-curv
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/placement_strong_hypotheses_v1/logs/curvature_%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/placement_strong_hypotheses_v1/logs/curvature_%j.err
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/zhihan/research/Basin_C1
/tmp/c1-transfer-audit-env/bin/python diagnostics/placement_strong_hypotheses_v1/run_curvature.py
