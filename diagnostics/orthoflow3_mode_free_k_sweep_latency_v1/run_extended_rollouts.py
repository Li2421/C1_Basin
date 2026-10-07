#!/usr/bin/env python3
"""Run only cache-missing draws under the identical prior physical controller."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path('/home/zhihan/research/Basin_C1')
OLD = ROOT / 'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('prior_runner', OLD / 'run_rollouts.py')
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
prior.OUT = Path(__file__).resolve().parent
prior.EXPERIMENT_UID = 'exp_mode_free_k_sweep_latency_v1'

if __name__ == '__main__': prior.main()
