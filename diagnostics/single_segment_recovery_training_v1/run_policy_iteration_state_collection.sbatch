#!/bin/bash
#SBATCH --job-name=ssrec_pi_states
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/pi-states-slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/pi-states-slurm-%j.err
set -euo pipefail
TASK_DIR=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:$TASK_DIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.08
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
mkdir -p "$TASK_DIR/logs"
exec "$PY" "$TASK_DIR/collect_policy_iteration_states.py" --device gpu
