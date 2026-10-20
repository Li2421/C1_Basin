"""Exact pure generator-pair training labels, with separate cached references."""
from common import *
from design import anchor,action_case

def main():
    assert (HERE/'training_COMPLETE.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');names=base['controller_order'];types=np.asarray([base['types'][n] for n in names]);nc=len(names)
    labels=np.full((nc,48,26,1),-2,np.int8);uids=np.full(labels.shape,'',dtype='U96');events={k:np.zeros_like(labels) for k in ('collision','timeout','unknown')}
    refs={k:np.full((nc,48,3,1),-2,np.int8) if k=='labels' else np.full((nc,48,3,1),'',dtype='U96') if k=='uids' else np.zeros((nc,48,3,1),np.int8) for k in ('labels','uids',*events)}
    with rollout_db.connect(True) as db:
        for c in read(HERE/'training_CASES.json'):
            i=names.index(c['base_controller_name']);s=c['state_index'];a=c['training_slot'];r=db.execute('SELECT success,collision,timeout,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None
            assert labels[i,s,a,0]==-2;labels[i,s,a,0]=-1 if r['numerical_failure'] else r['success'];uids[i,s,a,0]=c['rollout_uid']
            for k in events:events[k][i,s,a,0]=r['numerical_failure'] if k=='unknown' else r[k]
        # Read references only from exact previously cached identities, never submit.
        states=read(HERE/'TRAINING_STATES.json')
        legacy_refs=None
        with np.load(D/'crosstype1c_two_eta_schedule_v1/TRAINING_REFERENCES.npz',allow_pickle=False) as z:legacy_refs={k:z[k] for k in z.files}
        oldnames=read(D/'crosstype1c_two_eta_schedule_v1/BASE_PROTOCOL.json')['controller_order']
        for i,n in enumerate(names):
            for si,s in enumerate(states):
                for slot,arm in enumerate(('safety_baseline','S500','Sprog')):
                    if n in oldnames:
                        for k in refs:refs[k][i,si,slot,0]=legacy_refs[k][oldnames.index(n),si,slot,0]
                        continue
                    c,_=action_case(base,n,s,anchor(base,arm),0,si)
                    # Historical mixed-zoo human progress schedule used its own
                    # canonical profile key, though the physical execution is equal.
                    if arm=='Sprog':
                        import copy
                        from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
                        v=copy.deepcopy(base['profiles'][n]['payload']);v['progress_schedule']=dict(pstar=.9681620884830543,fallback=600,implementation_sha256=sha(HERE/'progress.py'))
                        cid=uid('ctl',v);c['rollout_uid']=uid('roll',dict(state=s['state_uid'],eta=eta_identity(base['eta'][13])[0],controller=cid,seed=canonical(dict(future_index=0))))
                    r=db.execute('SELECT success,collision,timeout,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                    if r is None:continue
                    refs['labels'][i,si,slot,0]=-1 if r['numerical_failure'] else r['success'];refs['uids'][i,si,slot,0]=c['rollout_uid']
                    for k in events:refs[k][i,si,slot,0]=r['numerical_failure'] if k=='unknown' else r[k]
    assert not np.any(labels==-2)
    with np.load(HERE/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:indices=z['candidate_indices']
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:h=z['h'][:48];c=z['c'][:48];action=z['action'][:48][np.arange(48)[None,:,None],indices];proposals=z['proposals'][:48];state_uids=z['state_uids'][:48]
    with np.load(HERE/'TRAINING_CONTROLLER_FEATURES.npz',allow_pickle=False) as z:assert np.array_equal(z['names'],names);features=z['features']
    np.savez_compressed(HERE/'TRAINING_BUNDLE.npz',names=names,types=types,labels=labels,uids=uids,features=features,h=h,c=c,action=action,candidate_indices=indices,proposals=proposals,state_uids=state_uids,**events)
    np.savez_compressed(HERE/'TRAINING_REFERENCES.npz',names=names,types=types,**refs)
    seal('TRAINING_BUNDLE.receipt.json',[HERE/'TRAINING_BUNDLE.npz',HERE/'TRAINING_REFERENCES.npz',HERE/'assemble.py',HERE/'training_COMPLETE.receipt.json',HERE/'TRAINING_CONTROLLER_FEATURES.receipt.json'],seed0_complete=True,critic_training_data_excludes_human_anchors=True,reference_file_not_training=True)
    print('104-controller pure two-eta training bundle sealed',flush=True)

def space():
    from families import family
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:types=z['types'];labels=z['labels'];indices=z['candidate_indices']
    diag=indices//40==indices//5%8;success=labels[...,0]==1;rows={}
    for name in [*sorted(set(types)),'combined','learned','optimization','search']:
        mask=np.asarray([name=='combined' or t==name or family(t)==name for t in types]);d=np.where(diag,success,False).max(-1).mean(-1);n=np.where(~diag,success,False).max(-1).mean(-1)
        def equal(v):return float(np.mean([v[mask&(types==t)].mean() for t in sorted(set(types[mask]))]))
        rows[name]=dict(controllers=int(mask.sum()),diagonal_in_sample_best_pct=equal(d)*100,non_diagonal_sampled_in_sample_best_pct=equal(n)*100)
    write(HERE/'SELECTABLE_SPACE.json',dict(utc=utc(),rows=rows,scope='Training state seed0 only;8 diagonals and18 sampled non-diagonals, not full288 oracle',evidence='DEVELOPMENT'))
    seal('SELECTABLE_SPACE.receipt.json',[HERE/'SELECTABLE_SPACE.json',HERE/'TRAINING_BUNDLE.receipt.json',HERE/'assemble.py'])

if __name__=='__main__':main() if len(sys.argv)==1 else space()
