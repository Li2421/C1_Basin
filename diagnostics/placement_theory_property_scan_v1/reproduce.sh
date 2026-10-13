#!/bin/bash
# Read-only reanalysis of saved original trajectories and frozen controller probes.
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
for script in analyze_actions.py analyze_margins.py analyze_root_refine.py \
  analyze_boundary.py analyze_basis_feedback.py analyze_global.py \
  analyze_geometry.py build_counterexamples.py plot_properties.py; do
  "$audit_python" "$audit_dir/$script" > "$audit_dir/${script%.py}.log"
done
"$audit_python" "$audit_dir/verify.py" > "$audit_dir/verification.log"
"$audit_python" "$audit_dir/make_artifact_index.py"
