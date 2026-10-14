#!/bin/bash
#SBATCH --job-name=r12_freeze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:15:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/controller_new_checkpoint_confirmation_v1/prepare_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
R12_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$R12_PY -B diagnostics/controller_new_checkpoint_confirmation_v1/probe.py merge
$R12_PY -B diagnostics/controller_new_checkpoint_confirmation_v1/models.py
$R12_PY -B diagnostics/controller_new_checkpoint_confirmation_v1/design.py create
