#!/bin/bash
#SBATCH --job-name=controller_q_round3_p2merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/controller_q_transfer_round3_v1/P2PRIME_MERGE.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_q_transfer_round3_v1/p2prime_run.py merge
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_q_transfer_round3_v1/p2prime_run.py postflight
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_q_transfer_round3_v1/p2prime_data.py
