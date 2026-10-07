#!/bin/bash
#SBATCH --job-name=c1-ff-small-eval
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/orthoflow3_field_transfer_small_v1/eval_%j.out
#SBATCH --error=diagnostics/orthoflow3_field_transfer_small_v1/eval_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_field_transfer_small_v1.study finalize
