#!/usr/bin/env bash
#SBATCH --job-name=oracle-bdry-256
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/oracle_boundary_confidence_audit/slurm-256-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/oracle_boundary_confidence_audit/slurm-256-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
OUT="$ROOT/diagnostics/oracle_boundary_confidence_audit"
export PYTHONPATH="$ROOT" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
"$PY" "$OUT/run_eta_zero.py" --arms to256_shard0_arms.json --stage to256_shard0 --device gpu --batch 32 &
pid0=$!
"$PY" "$OUT/run_eta_zero.py" --arms to256_shard1_arms.json --stage to256_shard1 --device gpu --batch 32 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
exit "$status"
