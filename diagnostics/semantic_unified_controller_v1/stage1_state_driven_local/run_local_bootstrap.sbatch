#!/bin/bash
#SBATCH --job-name=semantic_local0
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage1_state_driven_local/logs/local0-slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage1_state_driven_local/logs/local0-slurm-%j.err
set -euo pipefail
D=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage1_state_driven_local
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
M=$D/branch_manifests/local_bootstrap_safety.json
mkdir -p "$D/logs"
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1:$D
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.12
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0 "$PY" "$D/run_local_branches.py" --manifest "$M" \
    --shard-index "$shard" --shard-count 2 --device gpu \
    >"$D/logs/local0-shard${shard}.out" 2>"$D/logs/local0-shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
