#!/usr/bin/env bash
#SBATCH --job-name=gate-conf-256
#SBATCH --partition=gpu
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_gate_confidence_aware_v1/slurm-256-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_gate_confidence_aware_v1/slurm-256-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT="$ROOT/diagnostics/gphi_gate_confidence_aware_v1"
export PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.08
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
pids=()
for shard in 0 1 2 3; do
  "$PY" "$OUT/run_eta_zero.py" --arms "to256_shard${shard}_arms.json" --stage "to256_shard${shard}" --device gpu --batch 32 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
