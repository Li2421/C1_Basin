#!/bin/bash
#SBATCH --job-name=gap1_n20_competence_data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_collect_competence.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.collect_scale_competence \
  --agents 20 \
  --output diagnostics/gap_flow_scale_20261007/n20_competence_data_v2 \
  --train 48 --dev 12 \
  --modes solo five half parked_half full \
  --permutations 2 --seed 20261009
