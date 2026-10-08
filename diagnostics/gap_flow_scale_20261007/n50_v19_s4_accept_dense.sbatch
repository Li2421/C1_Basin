#!/bin/bash
#SBATCH --job-name=gap1_n50_v19_s4_dense
#SBATCH --partition=cpu
#SBATCH --array=0-23%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n50_v19_s4_accept_dense_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if (( SLURM_ARRAY_TASK_ID < 12 )); then
  mode=one_way
  case_index=$SLURM_ARRAY_TASK_ID
else
  mode=one_way_mirror
  case_index=$((SLURM_ARRAY_TASK_ID - 12))
fi
case_id=$(printf '%02d' "$case_index")
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.evaluate_safety_audit \
  --agents 50 --split dev --observation-version competence_v2 \
  --fresh-count 1 --fresh-seed 398201 --fresh-start "$case_index" \
  --goal-stop --seed 7404 \
  --checkpoint-override diagnostics/gap_flow_scale_20261007/n50_reflection_candidate_v19_s4/flow.pkl \
  --mode "$mode" \
  --output "diagnostics/gap_flow_scale_20261007/n50_v19_s4_accept_${mode}_i${case_id}"
