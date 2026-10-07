#!/bin/bash
#SBATCH --job-name=k16-A-freeze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%j_freeze.out
#SBATCH --error=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%j_freeze.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
audit_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$audit_python" diagnostics/orthoflow3_k16_critic_canonicalization_v1/test_phase_a.py
"$audit_python" diagnostics/orthoflow3_k16_critic_canonicalization_v1/review_phase_a.py
"$audit_python" diagnostics/orthoflow3_k16_critic_canonicalization_v1/confirmation_a.py freeze
