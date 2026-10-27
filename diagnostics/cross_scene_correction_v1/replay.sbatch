#!/bin/bash
#SBATCH --job-name=cross_scene_replay
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/zhihan/research/Basin_C1
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/cross_scene_correction_v1/replay_%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
exec strace -f -e trace=file -o /home/zhihan/research/Basin_C1/diagnostics/cross_scene_correction_v1/STRACE_REPLAY.log /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/cross_scene_correction_v1/history.py replay
