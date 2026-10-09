#!/bin/bash
#SBATCH --job-name=gap1_eta_v4_prefix32
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --output=datasets/gap1_eta_scaling_v4_global128/logs/%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
: "${GAP1_BATCH_OFFSET:?must be 0, 1000, or 2000}"
: "${GAP1_ETA_START:=0}"
if (( GAP1_ETA_START < 0 || GAP1_ETA_START > 96 || GAP1_ETA_START % 32 != 0 )); then
  echo "invalid 32-wide global eta block: $GAP1_ETA_START" >&2
  exit 2
fi
global_index=$((GAP1_BATCH_OFFSET + SLURM_ARRAY_TASK_ID))
if (( global_index < 0 || global_index >= 2560 )); then
  echo "invalid global task index: $global_index" >&2
  exit 2
fi
if (( global_index < 1280 )); then
  n=10
  local_index=$global_index
else
  n=20
  local_index=$((global_index - 1280))
fi
state_index=$((local_index / 32))
eta_index=$((GAP1_ETA_START + local_index % 32))
pair_index=$((state_index * 128 + eta_index))
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.run_gap1_eta \
  --design datasets/gap1_eta_scaling_v4_global128 \
  --agents "$n" --pair-index "$pair_index" --stage logical_robust
