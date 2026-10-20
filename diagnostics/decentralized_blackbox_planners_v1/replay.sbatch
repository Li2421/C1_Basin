#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --job-name=r30e_replay
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/decentralized_blackbox_planners_v1/replay_%j.log
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/decentralized_blackbox_planners_v1/replay.py round28
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/decentralized_blackbox_planners_v1/replay.py round33
