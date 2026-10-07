#!/bin/bash
#SBATCH --job-name=c1-simple-response
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/simple_response_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/simple_response_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.simple_response
