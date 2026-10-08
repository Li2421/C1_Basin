#!/bin/bash
#SBATCH --job-name=gap1_n20_recovery_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_audit_recovery_v4_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
base=diagnostics/gap_flow_scale_20261007
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$py" -m bottleneck_family.audit_n50_dataset \
  --dataset "$base/n20_competence_merged_v4/dataset" \
  --output "$base/n20_competence_merged_v4/phase_audit.json"
"$py" - <<'PY'
import json
from pathlib import Path

root = Path('diagnostics/gap_flow_scale_20261007')
original = json.loads((root / 'n20_competence_data_v2/dataset/manifest.json').read_text())
merged = json.loads((root / 'n20_competence_merged_v4/dataset/manifest.json').read_text())
phase = json.loads((root / 'n20_competence_merged_v4/phase_audit.json').read_text())['results']
dev_original = [row for row in original['files'] if row['split'] == 'dev']
dev_merged = [row for row in merged['files'] if row['split'] == 'dev']
assert dev_original == dev_merged, 'held-out DEV records changed'
assert merged['counts']['source']['early_queue_recovery'] > 0
assert merged['counts']['source']['terminal_multi_recovery'] > 0
for split in ('train', 'dev'):
    left, right = phase[f'{split}_LR'], phase[f'{split}_RL']
    assert left['trajectories'] == right['trajectories']
    assert left['transitions'] == right['transitions']
    for group in (left, right):
        assert group['complete_success_trajectories'] == group['trajectories']
        assert group['terminal_goal_occupancy_quantiles'] == [20, 20, 20]
print('N=20 recovery dataset passed complete-goal, direction-balance and untouched-DEV audit')
PY
