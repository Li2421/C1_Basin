#!/bin/bash
#SBATCH --job-name=recovery_safety
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/recovery_takeover_primitive_v1/logs/safety-slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/recovery_takeover_primitive_v1/logs/safety-slurm-%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/recovery_takeover_primitive_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/Basin_C1:/home/zhihan/research/02_C1_Toy_GiveWay
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.18
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/run_safety.py" --shard-index "$shard" --shard-count 2 --device gpu \
    >"$HERE/logs/safety-shard${shard}.out" 2>"$HERE/logs/safety-shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
