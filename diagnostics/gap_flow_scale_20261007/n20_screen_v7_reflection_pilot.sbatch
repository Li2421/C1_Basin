#!/bin/bash
#SBATCH --job-name=gap1_n20_reflect_screen
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_reflect_screen_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
candidate_index=$((SLURM_ARRAY_TASK_ID / 8))
local_index=$((SLURM_ARRAY_TASK_ID % 8))
mode_index=$((local_index / 4))
case_index=$((local_index % 4))
steps=(2000 6000 8000)
modes=(one_way one_way_mirror)
step=${steps[$candidate_index]}
mode=${modes[$mode_index]}
checkpoint="diagnostics/gap_flow_scale_20261007/n20_v7_reflect_s${step}/flow.pkl"
printf -v case_name '%04d' "$case_index"
output="diagnostics/gap_flow_scale_20261007/n20_v7_reflect_screen_s${step}/${mode}_${case_name}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.evaluate_safety_audit \
  --agents 20 --mode "$mode" --output "$output" \
  --checkpoint-override "$checkpoint" \
  --split dev --observation-version competence_v2 \
  --fresh-count 1 --fresh-start "$case_index" \
  --fresh-seed 162020 --seed 7404 --goal-stop
