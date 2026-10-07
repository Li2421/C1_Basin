#!/bin/bash
#SBATCH --job-name=c1-fresh-controller-test
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-6%6
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/fresh_test_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/fresh_test_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
if (( SLURM_ARRAY_TASK_ID == 0 )); then
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.fresh_controller_test features
else
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.fresh_controller_test run --index "$(((SLURM_ARRAY_TASK_ID - 1) * 2))" &
  rollout_worker_a=$!
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.fresh_controller_test run --index "$(((SLURM_ARRAY_TASK_ID - 1) * 2 + 1))" &
  rollout_worker_b=$!
  wait "$rollout_worker_a"
  wait "$rollout_worker_b"
fi
