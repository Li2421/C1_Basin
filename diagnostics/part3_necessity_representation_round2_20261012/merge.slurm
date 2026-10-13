#!/bin/bash
#SBATCH --job-name=part3r2_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/part3_necessity_representation_round2_20261012/merge.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_necessity_representation_round2_20261012/run.py merge
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/part3_necessity_representation_round2_20261012/run.py postflight
