#!/bin/bash
#SBATCH --job-name=r25_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/g1b_divergence_audit_v1/audit_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
for mode in old old32 new32 new64; do
 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/g1b_divergence_audit_v1/single.py "$mode"
done
