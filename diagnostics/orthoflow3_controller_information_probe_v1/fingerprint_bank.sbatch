#!/bin/bash
#SBATCH --job-name=c1-controller-bank
#SBATCH --partition=cpu
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/bank_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/bank_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
scenes=(double_bottleneck four_way_intersection ring_exchange)
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_information_probe_v1.fingerprint_bank \
  --scene "${scenes[$SLURM_ARRAY_TASK_ID]}"
