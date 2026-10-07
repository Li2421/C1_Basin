#!/bin/bash
#SBATCH --job-name=eta_final_wide6
#SBATCH --partition=gpu
#SBATCH --gres=shard:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_structured_eta_fresh_wide_v1/logs/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_structured_eta_fresh_wide_v1/logs/slurm-%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/gphi_structured_eta_fresh_wide_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
cd /home/zhihan/research/Basin_C1/toy_giveway
export PYTHONPATH=/home/zhihan/research/Basin_C1:/home/zhihan/research/02_C1_Toy_GiveWay
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
pids=()
starts=(0 34 68 101 134 167)
stops=(34 68 101 134 167 200)
for shard in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/run_evaluation.py" \
    --start-index "${starts[$shard]}" --stop-index "${stops[$shard]}" --device gpu \
    >"$HERE/logs/shard${shard}.out" 2>"$HERE/logs/shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
