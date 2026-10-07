#!/bin/bash
#SBATCH --job-name=rrisk-chunk-repl
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/rrisk_true_chunk/replication-%j.log

set -euo pipefail
cd /home/zhihan/research/C1_TRO_Rrisk
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/rrisk_true_chunk/run_true_chunk_validation.py \
  --stage replication \
  --pilot-analysis diagnostics/rrisk_true_chunk/data_v1/pilot_analysis/analysis.json \
  --pilot-records diagnostics/rrisk_true_chunk/data_v1/pilot/arm_records.json \
  --out diagnostics/rrisk_true_chunk/data_v1/replication
