#!/bin/bash
#SBATCH --job-name=c1-drop-rest-regress
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/drop_rest_regression_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/drop_rest_regression_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.motion_drop_rest_regression evaluate
