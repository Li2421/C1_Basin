#!/bin/bash
# Read-only analysis of already saved real rollout traces; no new simulation.
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
for script in analyze_panel.py analyze_sobol128.py analyze_grid9.py analyze_robust.py \
  analyze_hybrid.py analyze_hybrid_grid.py analyze_h1_gradients.py \
  analyze_h1_prediction_combined.py analyze_transfer_rank.py \
  analyze_curvature.py analyze_curvature_followup.py analyze_geometry_summary.py; do
  "$audit_python" "$audit_dir/$script" > "$audit_dir/${script%.py}.log"
done
"$audit_python" "$audit_dir/verify.py" > "$audit_dir/verification.log"
"$audit_python" "$audit_dir/make_artifact_index.py"
