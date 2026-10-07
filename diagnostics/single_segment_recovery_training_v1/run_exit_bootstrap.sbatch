#!/bin/bash
#SBATCH --job-name=ssrec_exit0
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/exit0-slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/exit0-slurm-%j.err
set -euo pipefail
TASK_DIR=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
MANIFEST="$TASK_DIR/branch_manifests/exit_bootstrap_never_exit.json"
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:$TASK_DIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.12
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
mkdir -p "$TASK_DIR/logs"
pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$TASK_DIR/run_paired_branches.py" \
    --manifest "$MANIFEST" --shard-index "$shard" --shard-count 2 --device gpu \
    >"$TASK_DIR/logs/exit0-shard${shard}.out" \
    2>"$TASK_DIR/logs/exit0-shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
