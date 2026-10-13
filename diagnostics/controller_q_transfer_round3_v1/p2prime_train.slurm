#!/bin/bash
#SBATCH --job-name=controller_q_round3_p2train
#SBATCH --partition=gpu
#SBATCH --gres=shard:8
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/controller_q_transfer_round3_v1/P2PRIME_TRAIN.log
set -euo pipefail
export JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_q_transfer_round3_v1/p2prime_train.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_q_transfer_round3_v1/p2prime_analyze.py
