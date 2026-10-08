#!/bin/bash
#SBATCH --job-name=gap1_n20_postgate_expert
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_postgate_recovery_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
: "${GAP1_CATEGORY:?set source category}"
: "${GAP1_TRACE_ROOT:?set completed train-only model trace directory}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.collect_n50_recovery_from_traces \
  --source diagnostics/gap_flow_scale_20261007/n20_competence_data_v2/dataset \
  --model-trace-dir "$GAP1_TRACE_ROOT/model_traces" \
  --output "diagnostics/gap_flow_scale_20261007/n20_postgate_recovery_phase_${GAP1_CATEGORY}" \
  --anchors 3000 5000 7000 9000 --seed 20261030 \
  --source-label late_postgate_recovery \
  --require-postgate --postgate-min-goal-distance 1.0 \
  --postgate-quantiles 0.2 0.5 0.8
