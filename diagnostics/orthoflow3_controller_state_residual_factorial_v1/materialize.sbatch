#!/bin/bash
#SBATCH --job-name=c1-factorial-data
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=diagnostics/orthoflow3_controller_state_residual_factorial_v1/materialize_%j.out
#SBATCH --error=diagnostics/orthoflow3_controller_state_residual_factorial_v1/materialize_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export JAX_PLATFORMS=cuda,cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_controller_state_residual_factorial_v1.pipeline postflight
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_controller_state_residual_factorial_v1.materialize
