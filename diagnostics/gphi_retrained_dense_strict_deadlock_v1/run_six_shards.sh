#!/usr/bin/env bash
#SBATCH --job-name=gphi-newdense6
#SBATCH --partition=gpu
#SBATCH --gres=shard:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

AUDIT_DIR=/home/zhihan/research/Basin_C1/diagnostics/gphi_retrained_dense_strict_deadlock_v1
PYTHON_BIN=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
cd "$AUDIT_DIR"

PIDS=()
for SHARD_INDEX in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.12 \
    "$PYTHON_BIN" run_dense.py --shard-index "$SHARD_INDEX" --shard-count 6 --device gpu --batch 32 \
    > "logs/shard${SHARD_INDEX}.out" 2> "logs/shard${SHARD_INDEX}.err" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=$?
done
exit "$STATUS"
