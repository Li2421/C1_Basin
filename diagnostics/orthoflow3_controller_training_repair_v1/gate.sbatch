#!/bin/bash
#SBATCH --job-name=c1-repair-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/orthoflow3_controller_training_repair_v1/gate_%j.out
#SBATCH --error=diagnostics/orthoflow3_controller_training_repair_v1/gate_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_controller_training_repair_v1.train gate
