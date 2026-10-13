#!/bin/bash
#SBATCH --job-name=c1-q-source-anchor-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/source_anchor_probe/merge_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -m diagnostics.transferable_state_conditioning_v1.source_probe merge
$PY -m diagnostics.transferable_state_conditioning_v1.source_probe signal
