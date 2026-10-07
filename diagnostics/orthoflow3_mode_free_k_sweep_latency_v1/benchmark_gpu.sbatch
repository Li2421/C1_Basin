#!/bin/bash
#SBATCH --job-name=ortho3-K-latency
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/logs/latency_%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/logs/latency_%j.err
set -euo pipefail
export JAX_PLATFORMS=cuda
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/benchmark_latency.py \
  --repeats 1000 --sequential-repeats 100 \
  --output diagnostics/orthoflow3_mode_free_k_sweep_latency_v1/latency_gpu.json
