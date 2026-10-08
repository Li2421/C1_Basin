#!/bin/bash
#SBATCH --job-name=gap1_n50_frozen_opposing
#SBATCH --partition=cpu
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n50_v19_frozen_opposing_test_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.evaluate_safety_audit \
  --agents 50 --mode opposing --split test \
  --observation-version competence_v2 --goal-stop --seed 7404 \
  --archive-start "$SLURM_ARRAY_TASK_ID" --archive-count 1 \
  --checkpoint-override diagnostics/gap_flow_scale_20261007/frozen/gap1_n50_flow_navigation_v1.pkl \
  --output "diagnostics/gap_flow_scale_20261007/n50_v19_frozen_opposing_test_i$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"
