#!/bin/bash
#SBATCH --job-name=c1-generalize
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=diagnostics/orthoflow3_c1_generalization_v1/interaction_%A_%a.out
#SBATCH --error=diagnostics/orthoflow3_c1_generalization_v1/interaction_%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
seed_list=(17 23 41)
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/orthoflow3_c1_generalization_v1/train_interaction.py --seed "${seed_list[$SLURM_ARRAY_TASK_ID]}"
