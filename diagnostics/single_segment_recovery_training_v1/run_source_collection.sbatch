#!/bin/bash
#SBATCH --job-name=ssrec_sources
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/source-collection-slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/source-collection-slurm-%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.12
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
mkdir -p "$HERE/logs"
pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/collect_safety_sources.py" \
    --shard-index "$shard" --shard-count 2 --device gpu \
    >"$HERE/logs/source-collection-shard${shard}.out" \
    2>"$HERE/logs/source-collection-shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
