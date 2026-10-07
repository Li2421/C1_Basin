#!/usr/bin/env bash
#SBATCH --job-name=k1train3
#SBATCH --partition=gpu
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-train-%j.out
#SBATCH --error=slurm-train-%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/gphi_dagger_k1_diagnostic_v1; PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python; cd "$HERE"; PIDS=()
for S in 17 23 41; do CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.20 "$PY" run_seed.py --seed "$S" >"logs/train${S}.out" 2>"logs/train${S}.err" & PIDS+=("$!"); done
R=0; for P in "${PIDS[@]}"; do wait "$P" || R=$?; done; exit "$R"
