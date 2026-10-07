#!/bin/bash
#SBATCH --job-name=c1-context-eval
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=diagnostics/orthoflow3_controller_context_loso_v1/%j_eval.out
#SBATCH --error=diagnostics/orthoflow3_controller_context_loso_v1/%j_eval.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export JAX_PLATFORMS=cpu
$PY -m diagnostics.orthoflow3_controller_context_loso_v1.evaluate target_context
export JAX_PLATFORMS=cuda,cpu
$PY -m diagnostics.orthoflow3_controller_context_loso_v1.evaluate source_audit
$PY -m diagnostics.orthoflow3_controller_context_loso_v1.evaluate predict
$PY -m diagnostics.orthoflow3_controller_context_loso_v1.evaluate evaluate
