#!/bin/bash
#SBATCH --job-name=c1-q-transfer-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/source_merge_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -m diagnostics.physical_state_critic_formal_v1.merge_loso merge
for sc in toy_give_way double_bottleneck four_way_intersection ring_exchange; do
  for split in train validation; do
    $PY -m diagnostics.physical_state_critic_formal_v1.merge_loso collect --scene "$sc" --split "$split"
  done
done
$PY -m diagnostics.transferable_state_conditioning_v1.audit_cache
