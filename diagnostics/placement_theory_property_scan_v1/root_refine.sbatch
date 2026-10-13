#!/bin/bash
#SBATCH --job-name=c1-property-root
#SBATCH --partition=cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/placement_theory_property_scan_v1/logs/root_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/placement_theory_property_scan_v1/logs/root_%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/zhihan/research/Basin_C1
/tmp/c1-transfer-audit-env/bin/python diagnostics/placement_theory_property_scan_v1/run_rollouts.py \
  --kind state_margin --protocol root_refine_protocol.json --preflight root_refine_cache_preflight.json \
  --shard "$SLURM_ARRAY_TASK_ID" --shards 4
