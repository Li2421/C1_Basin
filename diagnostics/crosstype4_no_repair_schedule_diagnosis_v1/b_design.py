"""Exact B training identities; old labels survive action-encoding changes."""
from common import *
from actions import case
from registry import register
from early_reuse import infer,validated

def training():
    prereg();assert (HERE/'A_DIAGNOSIS.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');catalogues=read(HERE/'CANDIDATES.json')['states'];names=base['controller_order'];rows=[];profiles={}
    with np.load(HERE/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:indices=z['candidate_indices'];assert list(z['names'])==names
    old=D/'crosstype3_unseen_controller_schedule_v1'
    with np.load(old/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:oldnames=list(z['names']);olduids=z['uids']
    for ni,name in enumerate(names):
        for si,s in enumerate(states):
            for slot,idx in enumerate(indices[ni,si]):
                a=catalogues[si]['actions'][int(idx)];c,cp=case(base,name,s,a,0,si,int(idx));c.update(training_slot=slot,phase_state_index=si)
                if name in oldnames and slot<26:assert c['rollout_uid']==str(olduids[oldnames.index(name),si,slot,0]),(name,si,slot)
                rows.append(c);profiles[cp['controller_uid']]=cp
    register('B_training',dict(base,states=states,phase='training'),rows,profiles)
    missing=read(HERE/'B_training_MISSING_CASES.json');assert len(missing)<=69312;aliases={};vt=validated()
    with rollout_db.connect(True) as db:
        for k,c in enumerate(missing):
            a=infer(c,base,db,vt)
            if a:aliases[c['rollout_uid']]=a
            if k%1000==0:print('B preflight',k,'/',len(missing),'local',len(aliases),flush=True)
    # Safety sources are deliberately scheduled first: actual zero-zero labels
    # can later be inspected as actual sources, never as local inferred chains.
    execution=sorted([c for c in missing if c['rollout_uid'] not in aliases],key=lambda c:(c['training_slot']!=26,c['base_controller_name'],c['state_index'],c['training_slot']))
    write(HERE/'B_training_LOCAL_EQUIVALENCE.json',aliases);write(HERE/'B_training_EXECUTION_CASES.json',execution);write(HERE/'B_training_BUDGET.json',dict(utc=utc(),requested=199104,exact_DB_present=199104-len(missing),raw_missing=len(missing),local_early_reuse=len(aliases),real_execution_upper_limit=len(execution),prior26_UID_checks=129792,old_rollouts_invalidated_by_action_encoding=False))
    seal('B_training_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('B_training_protocol.json','B_training_CASES.json','B_training_PREFLIGHT.receipt.json','B_training_LOCAL_EQUIVALENCE.json','B_training_EXECUTION_CASES.json','B_training_BUDGET.json','b_design.py','rollout_worker_v3.py','dynamic_reuse.py','actions.py','early_reuse.py','CANDIDATES.receipt.json','A_DIAGNOSIS.receipt.json')],before_submission=True)

def assemble():
    assert (HERE/'B_training_COMPLETE.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');names=base['controller_order'];shape=(122,48,34,1);labels=np.full(shape,-2,np.int8);uids=np.full(shape,'',dtype='U96');events={k:np.zeros(shape,np.int8) for k in ('collision','timeout','unknown')};aliases=read(HERE/'B_training_ALL_LOCAL_EQUIVALENCE.json');index={n:i for i,n in enumerate(names)}
    with rollout_db.connect(True) as db:
        for c in read(HERE/'B_training_CASES.json'):
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();r=dict(r) if r else aliases[c['rollout_uid']]['outcome'];ix=(index[c['base_controller_name']],c['state_index'],c['training_slot'],0);assert labels[ix]==-2;labels[ix]=-1 if r['numerical_failure'] else r['success'];uids[ix]=c['rollout_uid']
            for k in events:events[k][ix]=r['numerical_failure'] if k=='unknown' else r[k]
    assert not np.any(labels==-2)
    with np.load(HERE/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:indices=z['candidate_indices']
    with np.load(HERE/'STATE_INPUTS.npz',allow_pickle=False) as z:h=z['h'];context=z['c'];acts=z['action'][np.arange(48)[None,:,None],indices];proposals=z['proposals'];suids=z['state_uids']
    with np.load(HERE/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:assert list(z['names'])==names;features=z['features']
    np.savez_compressed(HERE/'TRAINING_BUNDLE.npz',names=names,types=[base['types'][n] for n in names],labels=labels,uids=uids,features=features,h=h,c=context,action=acts,candidate_indices=indices,proposals=proposals,state_uids=suids,**events)
    seal('TRAINING_BUNDLE.receipt.json',[HERE/n for n in ('TRAINING_BUNDLE.npz','B_training_COMPLETE.receipt.json','CANDIDATES.receipt.json','b_design.py')],complete_seed0=True,zero_is_only_safety_projection=True)
if __name__=='__main__':globals()[sys.argv[1]]()
