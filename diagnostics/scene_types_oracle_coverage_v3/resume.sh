#!/bin/bash
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1
if squeue -h -u "$USER" -o '%j' | rg -q '^stb3_pipeline$'; then
  echo 'Existing v3 pipeline is already active; no duplicate coordinator started.'
  exit 0
fi
exec srun -p cpu --job-name=stb3_pipeline --cpus-per-task=1 --mem=4800M --time=48:00:00 \
  env PYTHONDONTWRITEBYTECODE=1 \
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/scene_types_oracle_coverage_v3/main.py
