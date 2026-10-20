#!/bin/bash
#SBATCH --job-name=r28_expert
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=06:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/decentralized_pipeline_v1/r28_expert_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/decentralized_pipeline_v1/expert.py run
