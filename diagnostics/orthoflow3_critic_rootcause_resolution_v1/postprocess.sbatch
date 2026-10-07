#!/bin/bash
#SBATCH --job-name=c1-rootcause-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/postprocess_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/postprocess_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.postprocess "$1"
