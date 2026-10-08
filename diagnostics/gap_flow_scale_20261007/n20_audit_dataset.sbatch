#!/bin/bash
#SBATCH --job-name=gap1_n20_data_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_audit_dataset_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
base=diagnostics/gap_flow_scale_20261007/n20_competence_data_v2
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$py" -m bottleneck_family.audit_n50_dataset \
  --dataset "$base/dataset" --output "$base/phase_audit.json"
"$py" - <<'PY'
import json
from pathlib import Path

base = Path('diagnostics/gap_flow_scale_20261007/n20_competence_data_v2')
collection = json.loads((base / 'collection_report.json').read_text())
phase = json.loads((base / 'phase_audit.json').read_text())['results']
assert collection['agents'] == 20
assert not collection['teacher_failures'], collection['teacher_failures']
for split, expected in (('train', 480), ('dev', 120)):
    for direction in ('LR', 'RL'):
        group = phase[f'{split}_{direction}']
        assert group['trajectories'] == expected, (split, direction, group['trajectories'])
        assert group['complete_success_trajectories'] == expected
        assert group['terminal_goal_occupancy_quantiles'] == [20, 20, 20]
    assert phase[f'{split}_LR']['transitions'] == phase[f'{split}_RL']['transitions']
print('N=20 mirrored demonstration dataset passed complete-goal and direction-balance audit')
PY
