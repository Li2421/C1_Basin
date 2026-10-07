#!/bin/bash
#SBATCH --job-name=gphi_eta2
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/train_remaining_%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/train_remaining_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1/toy_giveway
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.10
CUDA_VISIBLE_DEVICES=0 "$PY" /home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/run_seed.py --seed 23 &
P0=$!
CUDA_VISIBLE_DEVICES=0 "$PY" /home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/run_seed.py --seed 41 &
P1=$!
S=0
wait "$P0" || S=$?
wait "$P1" || S=$?
exit "$S"
