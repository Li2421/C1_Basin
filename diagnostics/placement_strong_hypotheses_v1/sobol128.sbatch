#!/bin/bash
#SBATCH --job-name=c1-strong-qmc2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/placement_strong_hypotheses_v1/logs/qmc2_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/placement_strong_hypotheses_v1/logs/qmc2_%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/zhihan/research/Basin_C1
/tmp/c1-transfer-audit-env/bin/python diagnostics/placement_strong_hypotheses_v1/run_panel.py \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 6 --protocol sobol128_protocol.json \
  --preflight sobol128_cache_preflight.json --outdir sobol128_results
