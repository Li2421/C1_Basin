#!/usr/bin/env bash
#SBATCH --job-name=iwg-gate-w1
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/intervention_window_and_gate_v1/stage2_gate/slurm-worker1-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/intervention_window_and_gate_v1/stage2_gate/slurm-worker1-%j.err
set -euo pipefail
export PYTHONPATH=/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/intervention_window_and_gate_v1/stage2_gate/run_training_worker.py \
  --shard 1 --shards 2 --device gpu
