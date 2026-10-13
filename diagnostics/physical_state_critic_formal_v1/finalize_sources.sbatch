#!/bin/bash
#SBATCH --job-name=c1-q-source-finalize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/physical_state_critic_formal_v1/logs/finalize_sources_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$py -m diagnostics.physical_state_critic_formal_v1.merge_matched merge
$py -m diagnostics.physical_state_critic_formal_v1.merge_loso merge
for scene in toy_give_way ring_exchange; do
  for chain in TT FF; do
    for split in train validation; do
      $py -m diagnostics.physical_state_critic_formal_v1.merge_matched collect --scene "$scene" --chain "$chain" --split "$split"
    done
  done
done
for scene in toy_give_way double_bottleneck four_way_intersection ring_exchange; do
  for split in train validation; do
    $py -m diagnostics.physical_state_critic_formal_v1.merge_loso collect --scene "$scene" --split "$split"
  done
done
for panel in within_toy_TT within_toy_FF within_ring_TT within_db_FF within_four_FF loso_heldout_toy loso_heldout_db loso_heldout_four loso_heldout_ring; do
  $py -m diagnostics.physical_state_critic_formal_v1.assemble_trainval --panel "$panel"
done
fit_job=$(sbatch --parsable --array=0-53%2 diagnostics/physical_state_critic_formal_v1/train_unified.sbatch)
postfit_job=$(sbatch --parsable --dependency="afterok:$fit_job" --export=ALL,RING_FF_TEST_JOB="${RING_FF_TEST_JOB:?}" diagnostics/physical_state_critic_formal_v1/postfit.sbatch)
printf 'fit_job=%s\npostfit_job=%s\n' "$fit_job" "$postfit_job" > diagnostics/physical_state_critic_formal_v1/downstream_jobs.txt
