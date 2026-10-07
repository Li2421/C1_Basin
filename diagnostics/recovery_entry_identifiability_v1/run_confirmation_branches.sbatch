#!/bin/bash
#SBATCH --job-name=entry-id-confirm
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/recovery_entry_identifiability_v1/logs/confirm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/recovery_entry_identifiability_v1/logs/confirm-%j.err

set -euo pipefail

AUDIT=/home/zhihan/research/Basin_C1/diagnostics/recovery_entry_identifiability_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
RESULTS="$AUDIT/runs/paired_branches/confirmation"

mkdir -p "$RESULTS" "$AUDIT/logs"
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.12
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1

CUDA_VISIBLE_DEVICES=0 "$PY" "$AUDIT/run_paired_branches.py" \
  --manifest "$AUDIT/branch_seed_manifest_confirmation.json" \
  --result-directory "$RESULTS" \
  --shard-index 0 --shard-count 2 --device gpu \
  >"$AUDIT/logs/confirm-shard0.out" 2>"$AUDIT/logs/confirm-shard0.err" &
pid0=$!

# Slurm shards share one physical device; both workers use its visible index 0.
CUDA_VISIBLE_DEVICES=0 "$PY" "$AUDIT/run_paired_branches.py" \
  --manifest "$AUDIT/branch_seed_manifest_confirmation.json" \
  --result-directory "$RESULTS" \
  --shard-index 1 --shard-count 2 --device gpu \
  >"$AUDIT/logs/confirm-shard1.out" 2>"$AUDIT/logs/confirm-shard1.err" &
pid1=$!

wait "$pid0"
wait "$pid1"
