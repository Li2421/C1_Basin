#!/bin/bash
#SBATCH --job-name=semantic-stage2-src
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage2_recovery_entry/logs/source-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage2_recovery_entry/logs/source-%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.12
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1

HERE=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage2_recovery_entry
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
mkdir -p "$HERE/logs"
pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/collect_stage2_sources.py" \
    --shard-index "$shard" --shard-count 2 --device gpu \
    >"$HERE/logs/source-shard${shard}.out" 2>"$HERE/logs/source-shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
