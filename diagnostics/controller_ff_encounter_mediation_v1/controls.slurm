#!/bin/bash
#SBATCH --job-name=q_encounter_swap
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:15:00
#SBATCH --array=0-3%4
#SBATCH --output=diagnostics/controller_ff_encounter_mediation_v1/swap_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_ff_encounter_mediation_v1.controls infer --index "$SLURM_ARRAY_TASK_ID"
