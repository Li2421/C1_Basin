#!/bin/bash
#SBATCH --job-name=c1-motion-regression
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_regression_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_regression_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu
task_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.motion_counterexample_regression inputs
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.motion_counterexample_regression evaluate
