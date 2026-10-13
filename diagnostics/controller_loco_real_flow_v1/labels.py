"""Exact-UID label tensor after single-process DB merge and postflight."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
from shared_rollout_db.src.rollout_db import connect

def read(p):return json.loads(Path(p).read_text())

def main():
    assert read(HERE/'POSTFLIGHT.json')['missing']==0
    p=read(HERE/'protocol.json');names=p['controller_order'];states=p['states'];cases=read(HERE/'CASES.json')
    out=np.full((7,len(states),16,8),-2,np.int8)
    seen=set()
    with connect(True) as db:
        for r in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            assert row is not None and not row['conflict_quarantined']
            assert all(row[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
            key=(names.index(r['controller_name']),int(r['state_index'])+(128 if r['split']=='loco_test' else 0),int(r['eta_index']),int(r['seed']))
            assert key not in seen;seen.add(key)
            out[key]=-1 if row['numerical_failure'] else int(row['success'])
    assert len(seen)==len(cases)
    assert int((out[:,:128,:,:4]>=-1).sum())==len(cases)-7*64*16*8
    assert np.all(out[:,128:]>=-1)
    np.savez_compressed(HERE/'LABELS.npz',outcomes=out,controller_names=np.array(names),state_uids=np.array([s['state_uid'] for s in states]))
    audit=dict(status='COMPLETE EXACT UID LABEL TENSOR',shape=list(out.shape),success=int((out==1).sum()),failure=int((out==0).sum()),unknown=int((out==-1).sum()),unrequested=int((out==-2).sum()))
    (HERE/'LABEL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__':main()
