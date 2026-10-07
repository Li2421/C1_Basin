#!/bin/bash
#SBATCH --job-name=entry-id-probes
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/recovery_entry_identifiability_v1/logs/probes-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/recovery_entry_identifiability_v1/logs/probes-%j.err

set -euo pipefail
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/home/zhihan/research/Basin_C1/diagnostics/recovery_entry_identifiability_v1

D=/home/zhihan/research/Basin_C1/diagnostics/recovery_entry_identifiability_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" "$D/run_identifiability_probes.py"
