#!/bin/bash
#SBATCH --job-name=c1-minimal-cache-audit
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/minimal_cached_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/minimal_cached_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.minimal_response_diagnostic
