#!/usr/bin/env bash
set -euo pipefail

# Shared-machine policy: use one scheduler shard, avoid JAX's default full-device
# preallocation, and cap this tiny pilot to ten percent of device memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/gphi_pilot_training_v1/train_and_evaluate.py
