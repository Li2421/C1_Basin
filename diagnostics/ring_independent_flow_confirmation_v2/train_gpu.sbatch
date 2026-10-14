#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=r14a_repro
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/gpu_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_independent_flow_confirmation_v2/train_one.py base_u_v2_macflow 3104 reproduction_gpu
