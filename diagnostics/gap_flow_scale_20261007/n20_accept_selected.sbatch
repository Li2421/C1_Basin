#!/bin/bash
#SBATCH --job-name=gap1_n20_accept
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_accept_selected_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
: "${GAP1_SELECTED_CHECKPOINT:?set checkpoint selected only from non-opposing DEV}"
: "${GAP1_ACCEPT_OUTPUT:?set new candidate-specific acceptance output root}"
mode_index=$((SLURM_ARRAY_TASK_ID / 12))
case_index=$((SLURM_ARRAY_TASK_ID % 12))
if (( mode_index < 0 || mode_index >= 8 )); then
  echo "Invalid mode index: $mode_index" >&2
  exit 2
fi
modes=(single_even single_odd few_even few_odd one_way one_way_mirror \
       temporal_release_even_first temporal_release_odd_first)
mode=${modes[$mode_index]}
printf -v case_name '%04d' "$case_index"
output="${GAP1_ACCEPT_OUTPUT}/${mode}_${case_name}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.evaluate_safety_audit \
  --agents 20 --mode "$mode" --output "$output" \
  --checkpoint-override "$GAP1_SELECTED_CHECKPOINT" \
  --split dev --observation-version competence_v2 \
  --fresh-count 1 --fresh-start "$case_index" \
  --fresh-seed 398201 --seed 7405 --goal-stop
