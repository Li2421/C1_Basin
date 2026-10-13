#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/zhihan/research/Basin_C1
arms=(R0 R3 R4_action R4_openloop R3_R4_action)
fold=$((SLURM_ARRAY_TASK_ID / 5))
arm=${arms[$((SLURM_ARRAY_TASK_ID % 5))]}
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_loco_real_flow_v1/train.py "$fold" "$arm"
