#!/usr/bin/env python3
"""Reuse the exact prior controller/cache fingerprint protocol for new draws."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/home/zhihan/research/Basin_C1')
OLD = ROOT / 'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('prior_plan', OLD / 'prepare_cache_plan.py')
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
prior.OUT = Path(__file__).resolve().parent

if __name__ == '__main__':
    prior.main()
    path = prior.OUT / 'evaluation_protocol.json'
    obj = json.loads(path.read_text())
    obj['K'] = 16
    obj['new_sample_indices'] = list(range(4,16))
    obj['old_first_four_frozen_source'] = str(OLD / 'frozen_proposals.json')
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
