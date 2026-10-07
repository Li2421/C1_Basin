#!/bin/bash
#SBATCH --job-name=c1-diversity-test-input
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/diversity_test_inputs_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/diversity_test_inputs_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.diversity_confirmation physical
for input_profile in 0 1 2 3 4; do
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.diversity_confirmation features --index "$input_profile"
done
