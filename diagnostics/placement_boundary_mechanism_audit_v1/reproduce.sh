#!/usr/bin/env bash
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
"$audit_python" "$audit_dir/jacobian_probe.py"
"$audit_python" "$audit_dir/analyze_closed_loop.py" > "$audit_dir/trajectory_analysis.log"
"$audit_python" "$audit_dir/recurrence_spot_checks.py" > "$audit_dir/recurrence_spot_checks.log"
"$audit_python" "$audit_dir/analyze_boundary_chain.py" > "$audit_dir/boundary_chain_analysis.log"
"$audit_python" "$audit_dir/analyze_existing_labels.py" > "$audit_dir/existing_labels_analysis.log"
"$audit_python" "$audit_dir/analyze_flow_projection.py" > "$audit_dir/flow_projection_analysis.log"
"$audit_python" "$audit_dir/analyze_baseline_offsets.py" > "$audit_dir/baseline_offsets_analysis.log"
"$audit_python" "$audit_dir/plot_audit.py"
"$audit_python" "$audit_dir/verify.py" > "$audit_dir/verification.log"
"$audit_python" "$audit_dir/make_artifact_index.py"
