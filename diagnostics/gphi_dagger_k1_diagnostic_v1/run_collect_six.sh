#!/usr/bin/env bash
#SBATCH --job-name=k1collect6
#SBATCH --partition=gpu
#SBATCH --gres=shard:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-collect-%j.out
#SBATCH --error=slurm-collect-%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/gphi_dagger_k1_diagnostic_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
cd "$HERE"; PIDS=()
for I in 0 1 2 3 4 5; do
 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.12 "$PY" collect_k1.py --shard-index "$I" --shard-count 6 --batch 32 >"logs/collect${I}.out" 2>"logs/collect${I}.err" & PIDS+=("$!")
done
S=0; for P in "${PIDS[@]}"; do wait "$P" || S=$?; done; exit "$S"
