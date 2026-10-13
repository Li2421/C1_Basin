#!/bin/bash
#SBATCH --job-name=ring_c2_conditional
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/ring_state_selection_transfer_v1/c2_fallback.log
set -euo pipefail
export JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/zhihan/research/Basin_C1
if /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -c "import json; raise SystemExit(0 if not json.load(open('diagnostics/ring_state_selection_transfer_v1/c1/RESULTS.json'))['gate']['selection_transfer_qualified'] else 1)"; then
 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/ring_state_selection_transfer_v1/c2_fallback.py
else
 echo 'C1 passed: formal C2 budget and independent-state labels required; fallback not run.'
fi
