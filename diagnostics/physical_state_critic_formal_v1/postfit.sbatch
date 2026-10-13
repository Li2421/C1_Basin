#!/bin/bash
#SBATCH --job-name=c1-q-postfit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/physical_state_critic_formal_v1/logs/postfit_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
for panel in within_toy_TT within_toy_FF within_ring_TT within_db_FF within_four_FF loso_heldout_toy loso_heldout_db loso_heldout_four loso_heldout_ring; do
  $py -m diagnostics.physical_state_critic_formal_v1.train_unified freeze --panel "$panel"
done
$py -m diagnostics.physical_state_critic_formal_v1.predict_all freeze_all
for panel in within_toy_TT within_toy_FF within_ring_TT within_db_FF within_four_FF loso_heldout_toy loso_heldout_db loso_heldout_four loso_heldout_ring; do
  $py -m diagnostics.physical_state_critic_formal_v1.predict_all predict --panel "$panel"
done
$py -m diagnostics.physical_state_critic_formal_v1.predict_all bundle_loso
$py -m diagnostics.physical_state_critic_formal_v1.matched_worker freeze_remaining_test
matched_job=$(sbatch --parsable --array=0-815%4 diagnostics/physical_state_critic_formal_v1/matched_remaining_test.sbatch)
loso_job_a=$(sbatch --parsable --array=0-999%4 diagnostics/physical_state_critic_formal_v1/loso_test.sbatch)
loso_job_b=$(sbatch --parsable --export=ALL,ARRAY_OFFSET=1000 --array=0-87%4 diagnostics/physical_state_critic_formal_v1/loso_test.sbatch)
final_job=$(sbatch --parsable --dependency="afterok:$matched_job:$loso_job_a:$loso_job_b:${RING_FF_TEST_JOB:?}" diagnostics/physical_state_critic_formal_v1/finalize_tests.sbatch)
printf 'matched_test_job=%s\nloso_test_a=%s\nloso_test_b=%s\nfinalize_tests_job=%s\n' "$matched_job" "$loso_job_a" "$loso_job_b" "$final_job" >> diagnostics/physical_state_critic_formal_v1/downstream_jobs.txt
