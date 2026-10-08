#!/bin/bash
#SBATCH --job-name=gap1_n20_terminal_recovery
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_terminal_recovery_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.collect_n50_terminal_recovery \
  --source diagnostics/gap_flow_scale_20261007/n20_competence_data_v2/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_terminal_recovery_v1 \
  --variants 8 --pending-count 1 --minimum-goal-occupancy 19 \
  --max-displacement 1.4 --seed 20261009
