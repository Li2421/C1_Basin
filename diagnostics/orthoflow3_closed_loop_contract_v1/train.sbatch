#!/bin/bash
#SBATCH --job-name=contract-repair
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/orthoflow3_closed_loop_contract_v1/%j_train.out
#SBATCH --error=diagnostics/orthoflow3_closed_loop_contract_v1/%j_train.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
training_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$training_python -m diagnostics.orthoflow3_closed_loop_contract_v1.repair build
$training_python -m diagnostics.orthoflow3_closed_loop_contract_v1.repair generator
for seed in 17 23 41; do
  $training_python -m diagnostics.orthoflow3_closed_loop_contract_v1.repair critic --seed "$seed"
done
$training_python -m diagnostics.orthoflow3_closed_loop_contract_v1.repair freeze_critic
