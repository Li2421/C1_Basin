#!/bin/bash
#SBATCH --job-name=cq_cached_ff_fit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/controller_cached_ff_relearn_v1/logs/fit_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
kinds=(eta_only state_only current_field branch_nominal branch_field branch_field_contrast)
seeds=(17 23 41)
index=${SLURM_ARRAY_TASK_ID}
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -u -m diagnostics.controller_cached_ff_relearn_v1.learn fit --kind "${kinds[$((index/3))]}" --seed "${seeds[$((index%3))]}"
