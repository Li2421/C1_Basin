#!/bin/bash
#SBATCH --job-name=gap1_n20_opposing_recover
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_opposing_recover_v1_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
case_index=$SLURM_ARRAY_TASK_ID
if (( case_index < 12 || case_index >= 24 )); then
  echo "Invalid exact-index recovery case: $case_index" >&2
  exit 2
fi
printf -v case_name '%04d' "$case_index"
root=diagnostics/gap_flow_scale_20261007/n20_opposing_frozen_v1
output="$root/case_${case_name}"
archive="$root/interrupted_from_12828"
# Keep the preempted partial evidence; the evaluator requires a fresh folder.
if [[ -e "$output" ]]; then
  mkdir -p "$archive"
  if [[ -e "$archive/case_${case_name}" ]]; then
    echo "Refusing to overwrite archived partial case $case_name" >&2
    exit 2
  fi
  mv "$output" "$archive/case_${case_name}"
fi
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.evaluate_safety_audit \
  --agents 20 --mode opposing --output "$output" \
  --checkpoint-override diagnostics/gap_flow_scale_20261007/frozen/gap1_n20_flow_navigation_v1.pkl \
  --split test --observation-version competence_v2 \
  --fresh-count 1 --fresh-start "$case_index" \
  --fresh-seed 2026102012 --seed 7406 --goal-stop
