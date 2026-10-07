#!/usr/bin/env bash
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
if [[ ! -x "$audit_python" ]]; then
  echo '请将 AUDIT_PYTHON 设置为包含原项目依赖及 pandas、matplotlib 的 Python 路径。' >&2
  exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$audit_python" "$audit_dir/cache_analysis.py"
"$audit_python" "$audit_dir/state_chart_analysis.py"
"$audit_python" "$audit_dir/trajectory_linearization.py"
if [[ "${1:-}" == '--probes' ]]; then
  "$audit_python" "$audit_dir/local_probes.py"
  "$audit_python" "$audit_dir/targeted_checks.py"
  "$audit_python" "$audit_dir/inspect_runtime.py"
fi
"$audit_python" "$audit_dir/plot_audit.py"
"$audit_python" "$audit_dir/verify.py"
