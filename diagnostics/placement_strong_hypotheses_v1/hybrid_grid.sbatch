#!/bin/bash
#SBATCH --job-name=c1-strong-hygrid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/placement_strong_hypotheses_v1/logs/hygrid_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/placement_strong_hypotheses_v1/logs/hygrid_%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/zhihan/research/Basin_C1
/tmp/c1-transfer-audit-env/bin/python diagnostics/placement_strong_hypotheses_v1/run_hybrid.py \
  --protocol hybrid_grid_protocol.json --preflight hybrid_grid_cache_preflight.json \
  --outdir hybrid_grid_results --shard "$SLURM_ARRAY_TASK_ID" --shards 3
