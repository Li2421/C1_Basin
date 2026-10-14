#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=r14a_wide_repair
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/wide_repair_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_independent_flow_confirmation_v2/train_one.py base_u_plus_w_v4_wide_macflow 13114 repaired_new_flows/base_u_plus_w_v4_wide_macflow
