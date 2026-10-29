#!/bin/bash
set -euo pipefail
cd /home/zhihan/research/Basin_C1
if squeue -h -u "$USER" -o '%j' | rg -q '^brd1_pipeline$'; then
  echo 'Existing basis-reduction coordinator is active.'
  exit 0
fi
exec srun -p cpu --job-name=brd1_pipeline --cpus-per-task=1 --mem=6000M --time=72:00:00 \
 env PYTHONDONTWRITEBYTECODE=1 BRD_COORDINATOR=1 \
 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
 diagnostics/basis_reduction_within_four_v1/pipeline.py
