#!/bin/bash
#SBATCH --job-name=k16-Q-collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%j_collect.out
#SBATCH --error=diagnostics/orthoflow3_k16_critic_canonicalization_v1/logs/%j_collect.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
audit_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$audit_python" diagnostics/orthoflow3_k16_critic_canonicalization_v1/phase_a.py collect
"$audit_python" diagnostics/orthoflow3_k16_critic_canonicalization_v1/initial_expansion.py collect
"$audit_python" -c 'from diagnostics.orthoflow3_k16_critic_canonicalization_v1.train_phase_a import all_evidence; assert not any(r["collisions"] for r in all_evidence())'
