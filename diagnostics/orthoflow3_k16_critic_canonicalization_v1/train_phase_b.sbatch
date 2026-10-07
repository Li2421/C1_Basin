#!/bin/bash
#SBATCH --job-name=k16-phase-B
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%j_phase_b.out
#SBATCH --error=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%j_phase_b.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
training_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$training_python -m diagnostics.orthoflow3_k16_critic_canonicalization_v1.phase_b generator
for seed in 17 23 41; do
  $training_python -m diagnostics.orthoflow3_k16_critic_canonicalization_v1.phase_b critic --seed "$seed"
done
$training_python -m diagnostics.orthoflow3_k16_critic_canonicalization_v1.phase_b freeze_critic
