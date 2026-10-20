"""Use spare CPU slots during the other pilot's GPU training; no extra merger."""
from common import *
from collect39 import prepare
from submit_cpu40 import submit
from submit_stage import completed
import time

def main():
    while not (HERE/'train39_B_round_0_SUBMISSION.json').exists():time.sleep(5)
    assert all(r['passed'] for r in read(HERE/'EXPERT_GATE_RESULTS.json')['families'].values())
    stage='data39_F_round_0'
    if not (HERE/f'{stage}_CASES.json').exists():prepare('F',0)
    p=HERE/f'{stage}_SUBMISSION.json';job=read(p)['job'] if p.exists() else None
    while job is None:
        job=submit(stage,4,script='collect39.py')
        if job is None:time.sleep(5)
    while not completed(job):time.sleep(5)
    # The primary coordinator alone calls the official merger. Its intervening
    # experiment-level merges may incorporate these journals; never rerun them.
    write(HERE/'PILOT_F_INITIAL_COLLECTION_COMPLETE.json',dict(utc=utc(),job=job,stage=stage,merger='primary pilot coordinator only; no parallel merger',postflight='primary coordinator completes before F training'))

if __name__=='__main__':main()
