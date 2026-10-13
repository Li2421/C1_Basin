#!/bin/bash
#SBATCH --job-name=ring_c1_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/ring_state_selection_transfer_v1/c1/merge.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.ring_state_selection_transfer_v1.c1.run merge
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.ring_state_selection_transfer_v1.c1.analyze > diagnostics/ring_state_selection_transfer_v1/c1/analysis.log
