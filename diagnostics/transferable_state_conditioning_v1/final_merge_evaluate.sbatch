#!/bin/bash
#SBATCH --job-name=c1-q-final-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/final/logs/final_audit_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -m diagnostics.physical_state_critic_formal_v1.merge_loso merge
$PY -m diagnostics.transferable_state_conditioning_v1.final_worker merge_anchor
for bank in proposal anchor; do
  for scene in toy_give_way double_bottleneck four_way_intersection ring_exchange; do
    $PY -m diagnostics.transferable_state_conditioning_v1.evaluate_final collect \
      --scene "$scene" --bank "$bank"
  done
done
for panel in \
  within_toy_give_way within_double_bottleneck \
  within_four_way_intersection within_ring_exchange \
  loso_toy_give_way loso_double_bottleneck \
  loso_four_way_intersection loso_ring_exchange; do
  $PY -m diagnostics.transferable_state_conditioning_v1.evaluate_final proposal --panel "$panel"
  $PY -m diagnostics.transferable_state_conditioning_v1.evaluate_final anchor --panel "$panel"
done
$PY -m diagnostics.transferable_state_conditioning_v1.consolidate_final
