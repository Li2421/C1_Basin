"""Outcome-blind shuffled ownership; journals buffered without changed actions."""
from common import *
import argparse,numpy as np
import rollout
from shared_rollout_db.src.cache_writer import append_journal

def worker(index,workers,extra=False):
    d=HERE/'stage_2';cases=read(d/'MISSING_CASES.json');perm=np.random.default_rng(2026101317).permutation(len(cases))
    selected=[cases[int(i)] for i in perm[index::workers]]
    # Same executor and runtime; only append bookkeeping is buffered.
    buffer=[]
    def journal(records,exp,tag):
        buffer.extend(records)
        if len(buffer)>=16:append_journal(buffer.copy(),exp,tag);buffer.clear()
    rollout.append_journal=journal
    try:rollout.execute(d,selected,f'worker_{index}',extra)
    finally:
        if buffer:append_journal(buffer,read(d/'protocol.json')['experiment_uid'],f'worker_{index}')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--workers',type=int,default=24);ap.add_argument('--extra',action='store_true');a=ap.parse_args();worker(a.index,a.workers,a.extra)
