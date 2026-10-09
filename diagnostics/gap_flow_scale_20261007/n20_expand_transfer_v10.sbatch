#!/bin/bash
#SBATCH --job-name=gap1_n20_expand_v10
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_expand_v10_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
selection=diagnostics/gap_flow_scale_20261007/n20_v10_selection_pilot.json
mapfile -t top_steps < <(python3 - "$selection" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))['candidates'][:3]
if len(rows)!=3:
    raise SystemExit('pilot selector requires three candidates')
for row in rows:
    print(row['step'])
PY
)
candidate_index=$((SLURM_ARRAY_TASK_ID / 8))
local_index=$((SLURM_ARRAY_TASK_ID % 8))
mode_index=$((local_index / 2))
case_index=$((2 + local_index % 2))
if (( candidate_index < 0 || candidate_index >= 3 )); then
  echo "Invalid candidate index: $candidate_index" >&2
  exit 2
fi
steps=${top_steps[$candidate_index]}
printf -v checkpoint_name 'snapshot_%07d.pkl' "$steps"
checkpoint="diagnostics/gap_flow_scale_20261007/n20_train_n10_transfer_v10/$checkpoint_name"
modes=(single_even single_odd one_way one_way_mirror)
mode=${modes[$mode_index]}
printf -v case_name '%04d' "$case_index"
output="diagnostics/gap_flow_scale_20261007/n20_select_v10_s${steps}/${mode}_${case_name}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.evaluate_safety_audit \
  --agents 20 --mode "$mode" --output "$output" \
  --checkpoint-override "$checkpoint" \
  --split dev --observation-version competence_v2 \
  --fresh-count 1 --fresh-start "$case_index" \
  --fresh-seed 162020 --seed 7404 --goal-stop
