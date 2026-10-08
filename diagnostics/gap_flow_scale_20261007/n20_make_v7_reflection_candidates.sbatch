#!/bin/bash
#SBATCH --job-name=gap1_n20_reflect_v7
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_reflect_v7_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
dataset=diagnostics/gap_flow_scale_20261007/n20_competence_merged_v7/dataset
audit=diagnostics/gap_flow_scale_20261007/n20_mirror_audit_v7.json
train=diagnostics/gap_flow_scale_20261007/n20_train_postgate_v7
"$py" -m bottleneck_family.audit_n50_mirror_equivariance \
  --dataset "$dataset" \
  --checkpoint "$train/snapshot_0002000.pkl" \
  --checkpoint "$train/snapshot_0006000.pkl" \
  --checkpoint "$train/snapshot_0008000.pkl" \
  --output "$audit" --latents 1
for step in 2000 6000 8000; do
  printf -v checkpoint 'snapshot_%07d.pkl' "$step"
  "$py" -m bottleneck_family.make_n50_reflection_candidate \
    --dataset "$dataset" --source "$train/$checkpoint" \
    --output "diagnostics/gap_flow_scale_20261007/n20_v7_reflect_s${step}/flow.pkl" \
    --symmetry-audit "$audit"
done
