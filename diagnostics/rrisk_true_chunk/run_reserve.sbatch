#!/bin/bash
#SBATCH --job-name=rrisk-chunk-reserve
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/rrisk_true_chunk/reserve-%j.log

set -euo pipefail
cd /home/zhihan/research/C1_TRO_Rrisk
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/rrisk_true_chunk/run_true_chunk.py \
  --stage reserve \
  --out diagnostics/rrisk_true_chunk/data_v1/reserve
