#!/bin/bash
#SBATCH --job-name=state-eta-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_state_eta_interaction_v1/%j.out
#SBATCH --error=diagnostics/orthoflow3_state_eta_interaction_v1/%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -m diagnostics.orthoflow3_state_eta_interaction_v1.audit controls
$PY -m diagnostics.orthoflow3_state_eta_interaction_v1.audit interactions
