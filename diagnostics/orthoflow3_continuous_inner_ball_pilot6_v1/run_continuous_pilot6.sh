#!/bin/bash
#SBATCH --job-name=of3-cont-ball6
#SBATCH --partition=debug
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1/logs/run_%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1/logs/run_%j.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.20
cd /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python continuous_pilot6.py
