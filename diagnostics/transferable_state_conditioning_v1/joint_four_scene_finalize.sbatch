#!/bin/bash
#SBATCH --job-name=c1-q-joint-final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/joint_four_scene_v1/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$task_python -m diagnostics.physical_state_critic_formal_v1.merge_loso merge
$task_python -m diagnostics.transferable_state_conditioning_v1.joint_four_scene_evaluate evaluate_fresh
# Pure Q experiment: raw rollout and numerical evidence are the deliverables.
