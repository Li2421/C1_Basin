#!/bin/bash
#SBATCH --job-name=gap1_n20_merge_onset
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_merge_onset_v4_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.merge_n50_recovery \
  --source diagnostics/gap_flow_scale_20261007/n20_competence_merged_v3/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_onset_recovery_shard_0/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_onset_recovery_shard_1/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_onset_recovery_shard_2/dataset \
  --shard diagnostics/gap_flow_scale_20261007/n20_onset_recovery_shard_3/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_competence_merged_v4/dataset
