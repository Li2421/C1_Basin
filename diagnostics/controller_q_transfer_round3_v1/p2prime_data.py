"""Exact DB-backed P2′ tensors; missing and numerical unknown stay distinct."""
import json
import sys
from pathlib import Path

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
from shared_rollout_db.src.rollout_db import connect


def main():
    assert json.loads((HERE/'P2PRIME_POSTFLIGHT.json').read_text())['missing']==0
    protocol=json.loads((HERE/'P2PRIME_PROTOCOL.json').read_text());cases=json.loads((HERE/'P2PRIME_CASES.json').read_text())
    names=protocol['source_controllers']+protocol['held_controllers'];arr=np.full((7,352,16,8),-2,np.int8)
    with connect(True) as db:
        for r in cases:
            z=db.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            assert z is not None and not z['conflict_quarantined']
            ni=names.index(r['controller_name']);st={'train':0,'validation':256,'test':288}[r['split']]+r['state_index']
            assert arr[ni,st,r['eta_index'],r['seed']]==-2
            arr[ni,st,r['eta_index'],r['seed']]=-1 if z['numerical_failure'] else int(z['success'])
    assert (arr[:5,:256,:,:4]!=-2).sum()==40960
    assert (arr[:5,256:288,:,:4]!=-2).sum()==10240
    assert (arr[5:,288:,:,:]!=-2).sum()==16384
    np.savez_compressed(HERE/'P2PRIME_LABELS.npz',outcomes=arr,controller_names=np.array(names),
                        state_uids=np.array([s['state_uid'] for s in protocol['states']]))
    summary=dict(requested=len(cases),observed_valid=int(np.isin(arr,[0,1]).sum()),numerical_unknown=int((arr==-1).sum()),
                 unrequested_tensor_entries=int((arr==-2).sum()),source_controller_names=names[:5],held_controller_names=names[5:],
                 status='COMPLETE EXACT UID LABEL TENSOR')
    (HERE/'P2PRIME_LABEL_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
