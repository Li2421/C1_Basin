#!/bin/bash
#SBATCH --job-name=c1-ring-reversal-confirm
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_conditioning_probe_v1/%j_ring.out
#SBATCH --error=diagnostics/orthoflow3_controller_conditioning_probe_v1/%j_ring.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_controller_conditioning_probe_v1.ring_confirm run
