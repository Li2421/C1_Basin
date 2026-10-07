#!/bin/bash
#SBATCH --job-name=c1-phase-crossed-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/phase_factorial_summary_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/phase_factorial_summary_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
task_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.phase_factorial_train summarize
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.phase_factorial_analysis
