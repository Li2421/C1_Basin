#!/bin/bash
#SBATCH --job-name=c1-raw-held-merge
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/raw_merge_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/raw_merge_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
for action in postflight dataset evaluate; do
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.raw_bypass_confirmation "$action" --replicate "${1:?replica required}"
done
