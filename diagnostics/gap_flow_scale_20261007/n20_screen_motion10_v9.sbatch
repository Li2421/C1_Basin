#!/bin/bash
#SBATCH --job-name=gap1_n20_screen_v9
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_screen_v9_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
snapshot_index=$((SLURM_ARRAY_TASK_ID / 16))
local_index=$((SLURM_ARRAY_TASK_ID % 16))
mode_index=$((local_index / 4))
case_index=$((local_index % 4))
if (( snapshot_index < 0 || snapshot_index >= 8 )); then
  echo "Invalid snapshot index: $snapshot_index" >&2
  exit 2
fi
steps=$(((snapshot_index + 1) * 1000))
printf -v checkpoint_name 'snapshot_%07d.pkl' "$steps"
checkpoint="diagnostics/gap_flow_scale_20261007/n20_train_motion10_v9/$checkpoint_name"
modes=(single_even single_odd one_way one_way_mirror)
mode=${modes[$mode_index]}
printf -v case_name '%04d' "$case_index"
output="diagnostics/gap_flow_scale_20261007/n20_select_v9_s${steps}/${mode}_${case_name}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.evaluate_safety_audit \
  --agents 20 --mode "$mode" --output "$output" \
  --checkpoint-override "$checkpoint" \
  --split dev --observation-version competence_v2 \
  --fresh-count 1 --fresh-start "$case_index" \
  --fresh-seed 162020 --seed 7404 --goal-stop
