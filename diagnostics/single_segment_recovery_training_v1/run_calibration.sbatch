#!/bin/bash
#SBATCH --job-name=ssrec_cal
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/cal-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/cal-%j.err
set -euo pipefail
D=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1; PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python; M=$D/full_loop_manifests/calibration/selected_eta.json
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:$D XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.12 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
pids=(); for s in 0 1; do CUDA_VISIBLE_DEVICES=0 "$PY" "$D/run_full_episode_system.py" --manifest "$M" --shard-index "$s" --shard-count 2 --device gpu >"$D/logs/cal-shard${s}.out" 2>"$D/logs/cal-shard${s}.err" & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=$?; done; exit "$status"
