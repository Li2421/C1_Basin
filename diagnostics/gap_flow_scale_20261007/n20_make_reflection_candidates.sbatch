#!/bin/bash
#SBATCH --job-name=gap1_n20_reflect_make
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_reflect_make_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
dataset=diagnostics/gap_flow_scale_20261007/n20_competence_data_v2/dataset
audit=diagnostics/gap_flow_scale_20261007/n20_mirror_audit_v1.json
for step in 6000 20000; do
  printf -v checkpoint 'snapshot_%07d.pkl' "$step"
  "$py" -m bottleneck_family.make_n50_reflection_candidate \
    --dataset "$dataset" \
    --source "diagnostics/gap_flow_scale_20261007/n20_train_navigation_v1/$checkpoint" \
    --output "diagnostics/gap_flow_scale_20261007/n20_reflection_candidate_s${step}/flow.pkl" \
    --symmetry-audit "$audit"
done
