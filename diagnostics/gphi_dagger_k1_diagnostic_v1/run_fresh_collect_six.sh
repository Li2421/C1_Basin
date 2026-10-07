#!/usr/bin/env bash
#SBATCH --job-name=k1fresh6
#SBATCH --partition=gpu
#SBATCH --gres=shard:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-fresh-%j.out
#SBATCH --error=slurm-fresh-%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/gphi_dagger_k1_diagnostic_v1; PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python; cd "$HERE"; PIDS=()
for I in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.12 "$PY" collect_k1.py --cohort fresh6 --shard-index "$I" --shard-count 6 --batch 32 >"logs/fresh${I}.out" 2>"logs/fresh${I}.err" & PIDS+=("$!"); done
S=0; for P in "${PIDS[@]}"; do wait "$P" || S=$?; done; exit "$S"
