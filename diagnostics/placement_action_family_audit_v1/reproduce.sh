#!/usr/bin/env bash
set -euo pipefail
AUDIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_PYTHON="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$AUDIT_PYTHON" "$AUDIT_DIR/analyze.py" all
"$AUDIT_PYTHON" "$AUDIT_DIR/read_sequences.py"
"$AUDIT_PYTHON" "$AUDIT_DIR/verify.py"
"$AUDIT_PYTHON" "$AUDIT_DIR/plot.py"
"$AUDIT_PYTHON" "$AUDIT_DIR/write_report.py"
