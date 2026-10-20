from common import *
from zoo_design import profiles,case,LOCKED


def main():
    phases=('training_legacy','training_mixed','training_ppo')
    for phase in phases:assert (HERE/f'{phase}_COMPLETE.receipt.json').exists()
    base=profiles();names=[]
    for phase in phases:names.extend(read(HERE/f'{phase}_protocol.json')['controller_order'])
    assert len(names)==len(set(names)) and not any(base['types'][n] in LOCKED for n in names)
    states=read(HERE/'TRAINING_STATES.json');nc=len(names);labels=np.full((nc,48,16,2),-2,np.int8);uids=np.full(labels.shape,'',dtype='U96')
    events={k:np.zeros_like(labels) for k in ('collision','timeout','unknown')};indices={n:i for i,n in enumerate(names)}
    with rollout_db.connect(True) as db:
        for phase in phases:
            for c in read(HERE/f'{phase}_CASES.json'):
                i=indices[c['base_controller_name']];s=c['state_index'];a=c['candidate_index']
                for q in (0,1):
                    cc=c if q==0 else case(base,c['base_controller_name'],states[s],a,q)[0]
                    r=db.execute('SELECT success,numerical_failure,collision,timeout FROM rollout WHERE rollout_uid=?',(cc['rollout_uid'],)).fetchone()
                    if r is None:
                        assert q==1,'Unfinished seed0 is missing, never unknown';continue
                    labels[i,s,a,q]=-1 if r['numerical_failure'] else r['success'];uids[i,s,a,q]=cc['rollout_uid']
                    for e in events:events[e][i,s,a,q]=r['numerical_failure'] if e=='unknown' else r[e]
    assert not np.any(labels[...,0]==-2)
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:features={str(n):v for n,v in zip(z['names'],z['features'])}
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:h=z['h'][:48];c=z['c'][:48];actions=z['action'][:48]
    target=HERE/'TRAINING_BUNDLE.npz';assert not target.exists()
    np.savez_compressed(target,names=names,types=[base['types'][n] for n in names],labels=labels,uids=uids,features=[features[n] for n in names],h=h,c=c,action=actions,state_uids=[s['state_uid'] for s in states],**events)
    write(HERE/'BASE_PROTOCOL.json',{**base,'controller_order':names})
    seal('TRAINING_BUNDLE.receipt.json',[target,HERE/'BASE_PROTOCOL.json',HERE/'assemble_training.py',*[HERE/f'{p}_COMPLETE.receipt.json' for p in phases],HERE/'CONTROLLER_FEATURES.receipt.json'],seed0_complete=True,unknown_preserved=True,locked_outcomes_used=False)
    print('Training bundle complete:',nc,'controllers;',len(set(base['types'][n] for n in names)),'types')


if __name__=='__main__':main()
