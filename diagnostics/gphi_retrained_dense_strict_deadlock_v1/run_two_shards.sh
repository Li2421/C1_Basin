#!/usr/bin/env bash
#SBATCH --job-name=gphi-newdense
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

AUDIT_DIR=/home/zhihan/research/Basin_C1/diagnostics/gphi_retrained_dense_strict_deadlock_v1
PYTHON_BIN=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python

cd "$AUDIT_DIR"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" run_dense.py --shard-index 0 --shard-count 2 --device gpu --batch 32 > logs/shard0.out 2> logs/shard0.err &
PID0=$!
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" run_dense.py --shard-index 1 --shard-count 2 --device gpu --batch 32 > logs/shard1.out 2> logs/shard1.err &
PID1=$!

wait "$PID0"
wait "$PID1"
