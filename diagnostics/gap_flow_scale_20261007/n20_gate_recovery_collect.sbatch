#!/bin/bash
#SBATCH --job-name=gap1_n20_gate_recovery
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_gate_recovery_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
shard=$SLURM_ARRAY_TASK_ID
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.collect_n50_recovery_from_traces \
  --source diagnostics/gap_flow_scale_20261007/n20_competence_data_v2/dataset \
  --model-trace-dir "diagnostics/gap_flow_scale_20261007/n20_gate_model_shard_${shard}/model_traces" \
  --output "diagnostics/gap_flow_scale_20261007/n20_gate_recovery_shard_${shard}" \
  --anchors 1500 2500 3500 --source-label gate_local_recovery --seed 20261014
