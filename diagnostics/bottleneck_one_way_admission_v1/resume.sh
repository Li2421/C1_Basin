#!/bin/bash
set -euo pipefail
cd /home/zhihan/research/Basin_C1
if squeue -h -u "$USER" -o '%j' | rg -q '^owa1_pipeline$'; then
  echo 'Existing one-way admission coordinator is active.'
  exit 0
fi
exec srun -p cpu --job-name=owa1_pipeline --cpus-per-task=1 --mem=6000M --time=48:00:00 \
  env PYTHONDONTWRITEBYTECODE=1 OWA_COORDINATOR=1 \
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/bottleneck_one_way_admission_v1/main.py
