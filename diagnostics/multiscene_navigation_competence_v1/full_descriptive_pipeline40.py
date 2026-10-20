"""Frozen descriptive checks after navigation; no training or selection use."""
from common import *
from navigation_descriptive40 import prepare
from pilot_training40 import rollout_stage

def main():
    assert (HERE/'FULL_NAVIGATION_PHASE_COMPLETE.json').exists()
    plan=read(HERE/'FULL_DESCRIPTIVE_PLAN_FROZEN.json')
    receipt=read(HERE/'FULL_DESCRIPTIVE_PLAN_FROZEN.receipt.json')
    assert all(sha(p)==h for p,h in receipt['files'].items())
    for family in plan['families']:
        if not (HERE/f'NAVIGATION_CAPABILITY_{family}.json').exists():continue
        for role in plan['roles']:
            stage=f'descriptive40_{family}_{role}'
            if not (HERE/f'{stage}_CASES.json').exists():prepare(family,role)
            rollout_stage(stage,'descriptive_batch40.py')
    write(HERE/'FULL_DESCRIPTIVE_EXECUTION_COMPLETE.json',dict(utc=utc(),no_training_or_selection_use=True))

if __name__=='__main__':main()
