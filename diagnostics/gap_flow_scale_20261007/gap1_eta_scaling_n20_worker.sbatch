#!/bin/bash
#SBATCH --job-name=gap1_eta_n20_scale
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --output=datasets/gap1_eta_scaling_v3_n20/logs/%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
: "${GAP1_N:?set GAP1_N to 2, 10, or 20}"
: "${GAP1_STAGE:?set GAP1_STAGE to logical_robust or full_q16}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.run_gap1_eta \
  --design datasets/gap1_eta_scaling_v3_n20 \
  --agents "$GAP1_N" --pair-index "$SLURM_ARRAY_TASK_ID" \
  --stage "$GAP1_STAGE"
