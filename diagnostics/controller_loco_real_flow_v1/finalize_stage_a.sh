#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" diagnostics/controller_loco_real_flow_v1/run.py merge
"$PY" diagnostics/controller_loco_real_flow_v1/run.py postflight
"$PY" diagnostics/controller_loco_real_flow_v1/labels.py
"$PY" diagnostics/controller_loco_real_flow_v1/gate.py
"$PY" diagnostics/controller_loco_real_flow_v1/controller_q_decomposition.py
