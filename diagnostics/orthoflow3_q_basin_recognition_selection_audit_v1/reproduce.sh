#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_PY="${AUDIT_PY:-/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python}"
PLOT_PY="${PLOT_PY:-/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python}"
# export.py is optional on an archived copy: existing inputs are sufficient for analysis.
if [[ "${1:-}" == "--reexport" ]]; then "$AUDIT_PY" "$TASK_DIR/export.py"; fi
"$AUDIT_PY" "$TASK_DIR/test_metrics.py"
"$AUDIT_PY" "$TASK_DIR/analyze.py"
"$PLOT_PY" "$TASK_DIR/plot.py"
"$AUDIT_PY" "$TASK_DIR/verify.py"
"$AUDIT_PY" "$TASK_DIR/report.py"
