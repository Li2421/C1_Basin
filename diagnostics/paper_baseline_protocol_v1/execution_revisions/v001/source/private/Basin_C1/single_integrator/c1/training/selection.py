"""Validation selection follows min J_def subject to J_live <= epsilon."""
import math


def better_feasible_candidate(candidate, incumbent=None):
    for key in ('J_def', 'J_live', 'epsilon'):
        if not math.isfinite(candidate[key]):
            return False
    if candidate['J_live'] > candidate['epsilon']:
        return False
    return incumbent is None or candidate['J_def'] < incumbent['J_def']
