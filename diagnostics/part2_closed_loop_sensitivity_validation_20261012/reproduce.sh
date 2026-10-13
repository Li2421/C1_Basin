#!/usr/bin/env bash
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# The default regenerates outputs from existing cached data. Missing H80 files
# require a prior shared-cache preflight and an explicit --with-short-rollouts.
if [[ "${1:-}" == "--with-short-rollouts" ]]; then
  python3 - "$audit_dir" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]);pre=json.loads((p/'cache_preflight.json').read_text())
cost=json.loads((p/'cost_audit.json').read_text())
assert pre['summary']['ambiguous']==0 and cost['new_short_continuations']<=56
PY
  "$audit_python" "$audit_dir/trajectory_probe.py" run
fi

"$audit_python" "$audit_dir/reuse_analysis.py"
"$audit_python" "$audit_dir/boundary_supplement.py"
"$audit_python" "$audit_dir/derivative_validation.py"
"$audit_python" "$audit_dir/projection_candidate_probe.py"
"$audit_python" "$audit_dir/trajectory_probe.py" analyze
"$audit_python" "$audit_dir/recurrence_validation.py"
"$audit_python" "$audit_dir/gain_loss_mechanism.py"
python3 "$audit_dir/make_figures.py"
python3 "$audit_dir/finalize_artifacts.py"
