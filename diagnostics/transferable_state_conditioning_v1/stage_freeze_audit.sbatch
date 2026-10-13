#!/bin/bash
#SBATCH --job-name=c1-q-stage-loso
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/stage_v1/stage_audit_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TRANSFER_Q_WORK_ROOT=${TRANSFER_Q_WORK_ROOT:-/home/zhihan/research/Basin_C1/diagnostics/transferable_state_conditioning_v1/stage_v1}
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -m diagnostics.transferable_state_conditioning_v1.choose_final_models
$PY -m diagnostics.transferable_state_conditioning_v1.stage_evaluate_loso
