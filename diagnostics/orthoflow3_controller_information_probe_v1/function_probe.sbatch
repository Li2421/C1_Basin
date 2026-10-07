#!/bin/bash
#SBATCH --job-name=c1-controller-fnprobe
#SBATCH --partition=cpu
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/function_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/function_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu
scenes=(double_bottleneck four_way_intersection)
scene=${scenes[$SLURM_ARRAY_TASK_ID]}
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_intervention_generalization_v1.controller_signature \
  --scene "$scene" --shard 0 --shards 1
