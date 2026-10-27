#!/bin/bash
#SBATCH --job-name=blv2_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_distributed_v2/logs/analysis_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
unset XLA_FLAGS
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/baseline_literature_scenes_distributed_v2/analysis_jobs_cached.py worker "$BLV2_STAGE"
