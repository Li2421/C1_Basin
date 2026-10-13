#!/bin/bash
#SBATCH --job-name=c1-q-transfer-resid
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/residual_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
panels=(within_toy_give_way within_double_bottleneck within_four_way_intersection within_ring_exchange loso_toy_give_way loso_double_bottleneck loso_four_way_intersection loso_ring_exchange)
seeds=(17 23)
panel_index=$((SLURM_ARRAY_TASK_ID / 2))
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 2))]}
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.transferable_state_conditioning_v1.train_residual --panel "${panels[$panel_index]}" --seed "$seed"
