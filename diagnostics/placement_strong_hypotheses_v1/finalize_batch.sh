#!/bin/bash
# Formal batch completion must pass both numerical audit and required video entry.
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$audit_dir/reproduce.sh"
audit_python="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
"$audit_python" "$audit_dir/run_video_gate.py"
test "$(cat "$audit_dir/video_gate_exit_code.txt")" -eq 0
"$audit_python" -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.exit(0 if json.loads(p.read_text())["shared_cache_merge_complete"] else 1)' "$audit_dir/cache_integration_gap.json"
