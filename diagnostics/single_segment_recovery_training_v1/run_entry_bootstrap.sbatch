#!/bin/bash
#SBATCH --job-name=ssrec_entry0
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/entry0-slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/entry0-slurm-%j.err
set -euo pipefail
D=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1; PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:$D XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.12 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
pids=(); for s in 0 1; do CUDA_VISIBLE_DEVICES=0 "$PY" "$D/run_paired_branches.py" --manifest "$D/branch_manifests/entry_bootstrap_no_entry.json" --shard-index "$s" --shard-count 2 --device gpu >"$D/logs/entry0-shard${s}.out" 2>"$D/logs/entry0-shard${s}.err" & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=$?; done; exit "$status"
