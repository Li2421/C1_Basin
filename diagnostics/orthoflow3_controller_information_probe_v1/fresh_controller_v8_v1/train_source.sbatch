#!/bin/bash
#SBATCH --job-name=c1-ring-v8-source-fit
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/fresh_controller_v8_v1/train_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/fresh_controller_v8_v1/train_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
kinds=(fingerprint_eta fingerprint_full)
kind=${kinds[$SLURM_ARRAY_TASK_ID]}
for seed in 17 23 41; do
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
    -m diagnostics.orthoflow3_controller_information_probe_v1.probe train \
    --scene ring_exchange --kind "$kind" --seed "$seed" --crossmatrix
done
