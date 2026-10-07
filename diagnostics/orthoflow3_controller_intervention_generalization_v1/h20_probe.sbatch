#!/bin/bash
#SBATCH --job-name=c1-h20-intervention-probe
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_intervention_generalization_v1/h20_probe_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_controller_intervention_generalization_v1/h20_probe_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
kinds=(nominal20 rich20)
seeds=(17 23 41)
kind=${kinds[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_intervention_probe \
  train --kind "$kind" --seed "$seed"
