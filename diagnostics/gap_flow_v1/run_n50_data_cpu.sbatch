#!/usr/bin/env bash
#SBATCH --job-name=gap-n50-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/gap_flow_v1/logs/%j_n50_data.out
#SBATCH --error=diagnostics/gap_flow_v1/logs/%j_n50_data.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
flow_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
root=diagnostics/gap_flow_v1/n50_wide_recovery
mkdir -p "$root"
"$flow_python" -m bottleneck_family.flow_dataset --agents 50 --door-width 0.62 --train 24 --dev 8 --test 8 --recovery-anchors 3 --seed 20261007 --output "$root/dataset"
