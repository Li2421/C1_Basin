"""Exact DB labels, split into TRAIN and endpoint files; trainers read TRAIN only."""
from common import *
import numpy as np
from shared_rollout_db.src.rollout_db import connect

def main():
    d=HERE/'stage_2';p=read(d/'protocol.json');names=p['controller_order'];states=p['states'];n=p['train_states'];x=np.full((12,len(states),16,4),-2,dtype=np.int8);si={s['state_uid']:i for i,s in enumerate(states)}
    with connect(True) as db:
        for r in read(d/'CASES.json'):
            z=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone();assert z and not z['conflict_quarantined']
            assert all(z[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
            x[names.index(r['controller_name']),si[r['state_uid']],r['eta_index'],r['seed']]=-1 if z['numerical_failure'] else z['success']
    assert not (x[:,n:]==-2).any();assert np.all((x[:,:n]!=-2).sum((2,3))==32)
    np.savez_compressed(d/'TRAIN_LABELS.npz',outcomes=x[:,:n],controller_names=np.array(names),state_uids=np.array([s['state_uid'] for s in states[:n]]))
    np.savez_compressed(d/'TEST_DISCOVERY_LABELS.npz',outcomes=x[:,n:,:,:2],controller_names=np.array(names),state_uids=np.array([s['state_uid'] for s in states[n:]]))
    np.savez_compressed(d/'TEST_ENDPOINT_LABELS.npz',outcomes=x[:,n:,:,2:],controller_names=np.array(names),state_uids=np.array([s['state_uid'] for s in states[n:]]))
    write(d/'LABEL_AUDIT.json',dict(status='COMPLETE EXACT UID LABEL TENSOR',valid=int(((x==0)|(x==1)).sum()),unknown=int((x==-1).sum()),unrequested=int((x==-2).sum()),train_shape=list(x[:,:n].shape),discovery_seeds=[0,1],endpoint_seeds=[2,3],jax_enable_x64=False))
if __name__=='__main__':main()
