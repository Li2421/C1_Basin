#!/bin/bash
#SBATCH --job-name=r22_helper2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2500M
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_eta_pair_generality_v1/helper2_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_eta_pair_generality_v1/helper2.py 3
