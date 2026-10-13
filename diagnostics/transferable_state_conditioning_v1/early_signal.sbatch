#!/bin/bash
#SBATCH --job-name=c1-q-transfer-signal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/early_signal_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -m diagnostics.physical_state_critic_formal_v1.merge_loso merge
$PY -m diagnostics.transferable_state_conditioning_v1.audit_cache
# Slurm may reject a throttle update once the referenced array master task
# has completed; the cache merge/audit is the required part of this job.
scontrol update JobId=6895 ArrayTaskThrottle=4 || true
scontrol update JobId=6896 ArrayTaskThrottle=4 || true
