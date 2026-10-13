#!/bin/bash
#SBATCH --job-name=q_root_four_assemble
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=diagnostics/controller_four_scene_mechanism_v1/assemble.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_four_scene_mechanism_v1.study merge
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_four_scene_mechanism_v1.learn prepare
sbatch --parsable diagnostics/controller_four_scene_mechanism_v1/fit.slurm > diagnostics/controller_four_scene_mechanism_v1/fit_jobid.txt
