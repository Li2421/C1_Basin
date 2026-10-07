#!/bin/bash
#SBATCH --job-name=c1-tt-ff-context
#SBATCH --partition=debug
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_tt_ff_matched_learnability_v1/context_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_tt_ff_matched_learnability_v1/context_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_tt_ff_matched_learnability_v1.context smoke
fi
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_tt_ff_matched_learnability_v1.context run --worker "$SLURM_ARRAY_TASK_ID" --workers 5
