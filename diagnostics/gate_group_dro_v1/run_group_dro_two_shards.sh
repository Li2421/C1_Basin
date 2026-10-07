#!/usr/bin/env bash
#SBATCH --job-name=gate-group-dro
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gate_group_dro_v1/group-dro-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gate_group_dro_v1/group-dro-%j.err

set -euo pipefail
ROOT=/home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$ROOT"
"$PY" diagnostics/gate_group_dro_v1/run_group_dro_candidates.py "$@"
