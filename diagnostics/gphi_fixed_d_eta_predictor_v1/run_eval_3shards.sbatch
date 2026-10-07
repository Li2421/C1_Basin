#!/bin/bash
#SBATCH --job-name=gphi_eta_eval3
#SBATCH --partition=gpu
#SBATCH --gres=shard:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=43G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/logs/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/logs/slurm-%j.err
set -euo pipefail
HERE=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
cd /home/zhihan/research/Basin_C1/toy_giveway
export PYTHONPATH=/home/zhihan/research/Basin_C1:/home/zhihan/research/02_C1_Toy_GiveWay
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/run_structured_eta.py" --shard-index "$shard" --shard-count 3 --device gpu --batch 32 \
    >"$HERE/logs/shard${shard}.out" 2>"$HERE/logs/shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
