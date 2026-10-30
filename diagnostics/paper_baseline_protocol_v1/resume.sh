#!/bin/bash
set -euo pipefail
cd /home/zhihan/research/Basin_C1
if squeue -h -u "$USER" -o '%j' | rg -q '^pbp1_pipeline$'; then
  echo 'Existing paper-protocol coordinator active.'
  exit 0
fi
exec srun -p cpu --job-name=pbp1_pipeline --cpus-per-task=1 --mem=6000M --time=72:00:00 env PYTHONDONTWRITEBYTECODE=1 STB_COORDINATOR=1 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/paper_baseline_protocol_v1/pipeline.py
