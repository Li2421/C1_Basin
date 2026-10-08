#!/bin/bash
#SBATCH --job-name=gap1_n20_merge_postgate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_merge_postgate_v7_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.merge_n50_recovery \
  --source diagnostics/gap_flow_scale_20261007/n20_competence_merged_v5/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_postgate_recovery_phase_solo_LR/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_postgate_recovery_phase_solo_RL/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_postgate_recovery_phase_five_LR/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_postgate_recovery_phase_five_RL/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_postgate_recovery_phase_full_LR/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_postgate_recovery_phase_full_RL/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_competence_merged_v7/dataset
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.audit_n50_dataset \
  --dataset diagnostics/gap_flow_scale_20261007/n20_competence_merged_v7/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_competence_merged_v7/phase_audit.json
