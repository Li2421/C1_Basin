#!/usr/bin/env bash
#SBATCH --job-name=gphi-v4-train
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_pilot_training_v4/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_pilot_training_v4/slurm-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT="$ROOT/diagnostics/gphi_pilot_training_v4"
export PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
"$PY" "$OUT/train_seed.py" --seed 17 &
pid0=$!
"$PY" "$OUT/train_seed.py" --seed 23 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
"$PY" "$OUT/train_seed.py" --seed 41
"$PY" "$OUT/evaluate.py"
