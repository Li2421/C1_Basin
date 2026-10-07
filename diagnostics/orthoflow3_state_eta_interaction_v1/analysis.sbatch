#!/bin/bash
#SBATCH --job-name=interaction-analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_state_eta_interaction_v1/%j_analysis.out
#SBATCH --error=diagnostics/orthoflow3_state_eta_interaction_v1/%j_analysis.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -m diagnostics.orthoflow3_state_eta_interaction_v1.baselines
$PY -m diagnostics.orthoflow3_state_eta_interaction_v1.audit interactions
$PY -m diagnostics.orthoflow3_state_eta_interaction_v1.probe collect
$PY -m diagnostics.orthoflow3_state_eta_interaction_v1.analyze_probe
$PY -m diagnostics.orthoflow3_state_eta_interaction_v1.test_functional
