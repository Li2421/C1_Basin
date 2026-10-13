#!/bin/bash
#SBATCH --job-name=c1-q-transfer-assemble
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/source_assemble_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
for sc in toy_give_way double_bottleneck four_way_intersection ring_exchange; do
  $PY -m diagnostics.transferable_state_conditioning_v1.assemble_source --panel "within_$sc"
  $PY -m diagnostics.transferable_state_conditioning_v1.assemble_source --panel "loso_$sc"
done
