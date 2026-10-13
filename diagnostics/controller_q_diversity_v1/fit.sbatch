#!/bin/bash
#SBATCH --job-name=controller_q_fit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=diagnostics/controller_q_diversity_v1/logs/fit_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
kinds=(eta_only state_only context_only state_context field_only state_field state_field_compact state_contact state_contact_bilinear state_field_compact_bilinear)
seeds=(17 23 41)
index=${SLURM_ARRAY_TASK_ID}
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -u -m diagnostics.controller_q_diversity_v1.train fit --kind "${kinds[$((index/3))]}" --seed "${seeds[$((index%3))]}" --limit 128 --variant "${1:-source128}"
