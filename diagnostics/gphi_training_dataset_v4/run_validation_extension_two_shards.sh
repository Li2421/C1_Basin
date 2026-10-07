#!/usr/bin/env bash
#SBATCH --job-name=gphi-v4-valext
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v4/slurm-valext-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_training_dataset_v4/slurm-valext-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT="$ROOT/diagnostics/gphi_training_dataset_v4"
export PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
"$PY" "$OUT/add_validation_extensions.py"
"$PY" "$OUT/audit_candidates.py"
"$PY" "$OUT/adaptive_run.py" --arms eta_zero_valext_shard0_arms.json --stage eta_zero_valext_shard0 --device gpu --batch 2 &
pid0=$!
"$PY" "$OUT/adaptive_run.py" --arms eta_zero_valext_shard1_arms.json --stage eta_zero_valext_shard1 --device gpu --batch 2 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
exit "$status"
