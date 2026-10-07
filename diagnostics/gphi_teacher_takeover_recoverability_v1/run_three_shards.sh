#!/usr/bin/env bash
#SBATCH --job-name=gphi-takeover3
#SBATCH --partition=gpu
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=43G
#SBATCH --time=03:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_teacher_takeover_recoverability_v1/logs/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_teacher_takeover_recoverability_v1/logs/slurm-%j.err
set -euo pipefail

cd /home/zhihan/research/Basin_C1/toy_giveway
out=/home/zhihan/research/Basin_C1/diagnostics/gphi_teacher_takeover_recoverability_v1
venv=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

pids=()
for shard in 0 1 2; do
  "$venv" "$out/run_takeover.py" --shard-index "$shard" --shard-count 3 --device gpu --batch 32 \
    >"$out/logs/shard${shard}.out" 2>"$out/logs/shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
