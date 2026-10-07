"""Reuse the frozen Toy simulator/controller runner, writing only missing donor trials."""
import importlib.util
import sys
from pathlib import Path

ROOT=Path('/home/zhihan/research/Basin_C1')
OLD=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/run_rollouts.py'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('frozen_toy_runner',OLD)
runner=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=runner
spec.loader.exec_module(runner)
runner.OUT=Path(__file__).resolve().parent
runner.EXPERIMENT_UID='exp_core_hypothesis_generator_state_swap_v1'
if __name__=='__main__':runner.main()
