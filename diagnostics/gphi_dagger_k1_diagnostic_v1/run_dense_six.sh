#!/usr/bin/env bash
#SBATCH --job-name=k1dense6
#SBATCH --partition=gpu
#SBATCH --gres=shard:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=00:25:00
#SBATCH --output=slurm-dense-%j.out
#SBATCH --error=slurm-dense-%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/gphi_dagger_k1_diagnostic_v1; PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python; cd "$HERE"; PIDS=()
for I in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.12 "$PY" run_dense_eval.py --shard-index "$I" --shard-count 6 --device gpu --batch 32 >"logs/dense${I}.out" 2>"logs/dense${I}.err" & PIDS+=("$!"); done
S=0; for P in "${PIDS[@]}"; do wait "$P" || S=$?; done; exit "$S"
