#!/bin/bash
#SBATCH --job-name=c1-controller-bank-heldout
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/bankheld_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/bankheld_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
scenes=(double_bottleneck four_way_intersection ring_exchange)
seeds=(17 23 41)
scene=${scenes[$SLURM_ARRAY_TASK_ID]}
for kind in fingerprint_eta fingerprint_full; do
  for seed in "${seeds[@]}"; do
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
      -m diagnostics.orthoflow3_controller_information_probe_v1.probe train \
      --scene "$scene" --kind "$kind" --seed "$seed" --heldout-controller 2
  done
done
