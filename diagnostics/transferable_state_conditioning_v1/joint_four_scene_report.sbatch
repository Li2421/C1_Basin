#!/bin/bash
#SBATCH --job-name=c1-q-joint-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/joint_four_scene_v1/report_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.transferable_state_conditioning_v1.joint_four_scene_report
