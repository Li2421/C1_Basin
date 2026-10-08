#!/bin/bash
#SBATCH --job-name=gap1_n50_v13s1_sparse
#SBATCH --partition=cpu
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n50_v13s1_accept_sparse_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
case "$SLURM_ARRAY_TASK_ID" in
  0) mode=single_even;;
  1) mode=single_odd;;
  2) mode=few_even;;
  3) mode=few_odd;;
esac
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.evaluate_safety_audit \
  --agents 50 --split dev --observation-version competence_v2 \
  --fresh-count 12 --fresh-seed 203879 --goal-stop --seed 7403 \
  --checkpoint-override diagnostics/gap_flow_scale_20261007/n50_reflection_candidate_v13_s1/flow.pkl \
  --mode "$mode" \
  --output "diagnostics/gap_flow_scale_20261007/n50_v13s1_accept_${mode}"
