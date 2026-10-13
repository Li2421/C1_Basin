#!/usr/bin/env bash
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Read-only reuse and arithmetic analysis of existing 49,152 rollout and old
# first-action/Jacobian caches. No model forward, simulator step or training.
"$audit_python" "$audit_dir/analyze_geometry.py"
"$audit_python" "$audit_dir/trajectory_geometry.py"
"$audit_python" "$audit_dir/validate_geometry.py"
"$audit_python" "$audit_dir/make_figures.py"
"$audit_python" "$audit_dir/verify_cached.py"

# flow_substeps.py is provenance for 24 action-only traces already saved before
# the no-new-experiments instruction. It is intentionally excluded here.
