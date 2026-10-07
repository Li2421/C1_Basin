#!/bin/bash
#SBATCH --job-name=forced-eta-exit
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/forced_initial_eta_with_exit_v1/logs/run-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/forced_initial_eta_with_exit_v1/logs/run-%j.err

set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/forced_initial_eta_with_exit_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
mkdir -p "$HERE/logs"
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.12
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1

pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/run_audit.py" \
    --shard-index "$shard" --shard-count 2 --device gpu \
    >"$HERE/logs/shard${shard}.out" 2>"$HERE/logs/shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
