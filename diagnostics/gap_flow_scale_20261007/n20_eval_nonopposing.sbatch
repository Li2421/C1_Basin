#!/bin/bash
#SBATCH --job-name=gap1_n20_nonopposing
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_eval_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
: "${GAP1_CHECKPOINT:?set the N=20 candidate checkpoint}"
: "${GAP1_TAG:?set an output tag}"
: "${GAP1_FRESH_SEED:?set a fresh non-opposing physical-state seed}"
: "${GAP1_CASES_PER_MODE:?set the number of independent states per mode}"
mode_index=$((SLURM_ARRAY_TASK_ID / GAP1_CASES_PER_MODE))
case_index=$((SLURM_ARRAY_TASK_ID % GAP1_CASES_PER_MODE))
modes=(single_even single_odd few_even few_odd one_way one_way_mirror \
       temporal_release_even_first temporal_release_odd_first)
if (( mode_index < 0 || mode_index >= ${#modes[@]} )); then
  echo "Invalid mode index: $mode_index" >&2
  exit 2
fi
mode=${modes[$mode_index]}
printf -v case_name '%04d' "$case_index"
output="diagnostics/gap_flow_scale_20261007/${GAP1_TAG}/${mode}_${case_name}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.evaluate_safety_audit \
  --agents 20 --mode "$mode" --output "$output" \
  --checkpoint-override "$GAP1_CHECKPOINT" \
  --split dev --observation-version competence_v2 \
  --fresh-count 1 --fresh-start "$case_index" \
  --fresh-seed "$GAP1_FRESH_SEED" --seed 7404 --goal-stop
