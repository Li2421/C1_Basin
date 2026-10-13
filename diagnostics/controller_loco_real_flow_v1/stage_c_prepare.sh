#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
BASE=diagnostics/controller_loco_real_flow_v1
"$PY" "$BASE/stage_c_design.py" create
"$PY" -m shared_rollout_db.plan --manifest "$BASE/stage_c/PLANNED_ROLLOUTS.json" --output "$BASE/stage_c/CACHE_PREFLIGHT.json" > "$BASE/stage_c/PREFLIGHT_STDOUT.json"
"$PY" "$BASE/stage_c_design.py" inventory
"$PY" "$BASE/stage_c_design.py" register
"$PY" - <<'PY'
import json,hashlib,datetime
from pathlib import Path
p=Path('diagnostics/controller_loco_real_flow_v1/stage_c');a={'utc_time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':hashlib.sha256((p/'protocol.json').read_bytes()).hexdigest(),'prereg_sha256':hashlib.sha256((p.parent/'PREREGISTRATION_STAGE_C_AFTER_B.md').read_bytes()).hexdigest(),'preflight_summary':json.loads((p/'CACHE_PREFLIGHT.json').read_text())['summary'],'db_experiment_registered':True}
(p/'PILOT_OR_FULL_AUTHORITY.json').write_text(json.dumps(a,indent=2)+'\n')
print(a)
PY
