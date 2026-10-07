#!/bin/bash
#SBATCH --job-name=gphi_eta3
#SBATCH --partition=gpu
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=43G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/train_%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/train_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1/toy_giveway
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.10
export CUDA_VISIBLE_DEVICES=0
srun --exclusive --gres=shard:1 --cpus-per-task=2 --mem=13G \
  "$PY" /home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/run_seed.py --seed 17 &
srun --exclusive --gres=shard:1 --cpus-per-task=2 --mem=13G \
  "$PY" /home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/run_seed.py --seed 23 &
srun --exclusive --gres=shard:1 --cpus-per-task=2 --mem=13G \
  "$PY" /home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/run_seed.py --seed 41 &
wait
