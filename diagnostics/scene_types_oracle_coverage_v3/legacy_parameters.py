"""Read-only historical B3 points used solely for integration regression."""
from core import *
def regression_specs():
    p=ROOT/'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json'
    v=read(p)['lit_swap8_free']
    return [dict(eta=v[f'eta{i}']['eta_a'],coefficients={})for i in range(16)]
