#!/bin/bash
#SBATCH --job-name=c1-ring-controller-crossmatrix
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-3%1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/crossmatrix_ring_v1/run_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/crossmatrix_ring_v1/run_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_information_probe_v1.crossmatrix_ring \
  run --shard "$SLURM_ARRAY_TASK_ID" --shards 4
