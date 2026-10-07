#!/bin/bash
#SBATCH --job-name=db_mode_post
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_db_shared_mode_transfer_v1/logs/post_matrix_%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_db_shared_mode_transfer_v1/logs/post_matrix_%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_db_shared_mode_transfer_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python3.11
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
cd "$HERE"
"$PY" aggregate_matrix.py
"$PY" train_selector.py
