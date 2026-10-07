"""Run only preflight-missing frozen Toy replication requests.

Imports the previously validated rollout executor unchanged in its physics
loop. The sole shared-runner adjustment is metadata-only: state_id uses the
already-frozen source_group, avoiding alias collision with the old cohort.
"""

import importlib.util
import sys
from pathlib import Path

ROOT=Path('/home/zhihan/research/Basin_C1')
SRC=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/run_rollouts.py'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('frozen_toy_rollout_executor',SRC)
prior=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=prior
spec.loader.exec_module(prior)
prior.OUT=Path(__file__).resolve().parent/'toy_replication'
prior.EXPERIMENT_UID='exp_orthoflow3_c1_generalization_toy_replication_v1'

if __name__=='__main__':prior.main()
