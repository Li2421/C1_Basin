#!/usr/bin/env bash
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
DIR=diagnostics/controller_flow_blend_v1
"$PY" "$DIR/rollout.py" merge --stage stage_2
"$PY" "$DIR/rollout.py" postflight --stage stage_2
"$PY" "$DIR/stage2_labels.py"
