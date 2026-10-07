#!/bin/bash
#SBATCH --job-name=c1-function-target
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/function_confirm_flow_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/function_confirm_flow_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu
task_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
if "$task_python" -c "import json,sys; sys.exit(0 if json.load(open('diagnostics/orthoflow3_critic_rootcause_resolution_v1/controller_function_support/source_gate.json'))['passed'] else 1)"; then
  "$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.function_confirmation flow --replicate "$SLURM_ARRAY_TASK_ID"
fi
