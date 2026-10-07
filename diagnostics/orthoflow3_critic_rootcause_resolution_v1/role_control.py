"""Matched controller-slot information ablation, source families only.

This is an identifiability diagnostic, NOT a proposed cross-scene deployment
representation. The frozen joint Flow has ordered controller input/output slots.
Both arms have identical parameter counts, initialization, batches, objective,
and steps; only the added slot channels are zero versus their actual values.
No new seed-replication outcomes are used for fitting or checkpoint selection.
"""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
from diagnostics.orthoflow3_state_context_learning_audit_v1 import experiment as old

OUT=Path(__file__).resolve().parent/'role_control'
SRC=old.OUT
KINDS=('wide_db_role_zero','wide_db_role_aware')

def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Role control already frozen')
    OUT.mkdir(parents=True)
    for name in ('source_data_db.npz','source_entities.npz','source_states.json'):
        shutil.copyfile(SRC/name,OUT/name)
    p=copy.deepcopy(old.read(SRC/'protocol.json'))
    p.update(variants=KINDS,primary_question='Does imposed agent permutation invariance discard frozen Flow slot-response information?',
        comparator='Matched architecture with identical extra input dimensions: zeros vs true native controller slot one-hots',
        caveat='Source-only representation diagnostic; not cross-scene evidence and not a mode identifier',
        role_channels=dict(agents=4,pairs=8),normalization='unchanged historical FIT-only raw scaling; roles exact 0/1',
        training_data_sha256=old.sha(SRC/'source_data_db.npz'),
        fresh_seed_replication_labels_used=False,target_labels_used=False,new_rollout=0)
    old.write(OUT/'protocol.json',p)

def add_roles(x,enabled):
    x={k:v.copy() for k,v in x.items()};batch,n=x['agents'].shape[:2]
    assert n==4
    eye=np.eye(n,dtype=np.float32)*int(enabled)
    own=np.broadcast_to(eye,(batch,n,n))
    sender=np.broadcast_to(eye[None,:,None,:],(batch,n,n,n))
    receiver=np.broadcast_to(eye[None,None,:,:],(batch,n,n,n))
    x['agents']=np.concatenate([x['agents'],own],axis=-1)
    x['pairs']=np.concatenate([x['pairs'],sender,receiver],axis=-1)
    return x

def train(index):
    original_load=old.load
    def load(kind,fold):
        # Use byte-identical archived data, never the evolving global DB.
        d,x,fit,inner,test,norm=original_load('wide_db_full_raw',fold)
        x=add_roles(x,kind.endswith('role_aware'))
        norm['controller_slot_channels']=bool(kind.endswith('role_aware'))
        return d,x,fit,inner,test,norm
    old.OUT=OUT;old.KINDS=KINDS;old.load=load
    old.train(index)

def summarize():
    old.OUT=OUT;old.KINDS=KINDS;old.summarize()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','train','summarize'))
    p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='train':train(a.index)
    else:summarize()
