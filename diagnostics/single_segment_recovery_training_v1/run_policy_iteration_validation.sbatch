#!/bin/bash
#SBATCH --job-name=ssrec_pi_val
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:35:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/pi-val-slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/pi-val-slurm-%j.err
set -euo pipefail
D=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:$D
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.12
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
pids=()
for shard in 0 1; do
  (
    for manifest in "$D"/full_loop_manifests/validation_policy_iteration/*.json; do
      CUDA_VISIBLE_DEVICES=0 "$PY" "$D/run_full_episode_system.py" \
        --manifest "$manifest" --shard-index "$shard" --shard-count 2 --device gpu
    done
  ) >"$D/logs/pi-val-shard${shard}.out" 2>"$D/logs/pi-val-shard${shard}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
