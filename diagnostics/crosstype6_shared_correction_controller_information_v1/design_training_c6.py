"""Retain all old44 labels and add only nine preregistered q0 schedule labels."""
from common import *
from actions_c6 import action,case
from schedule_c6 import THRESHOLDS,TRIGGERS,encode
from registry import register
from reuse_c6 import infer

def seed_for(name,state_uid):
    return int.from_bytes(hashlib.sha256(json.dumps(['crosstype6-training-actions',2026102061,name,state_uid],separators=(',',':')).encode()).digest()[:8],'little')

def draws(name,s,proposals):
    seed=seed_for(name,s['state_uid']);rng=np.random.default_rng(seed);offset=int(seed%5);actions=[];sources=['requery','given','given']+['requery','given']*3;triggers=['none']*3+[t for t in list(TRIGGERS)[1:] for _ in range(2)]
    for i,(source,trigger) in enumerate(zip(sources,triggers)):
        early=int(rng.integers(8));late=int(rng.integers(8)) if source=='given' else 0;theta=THRESHOLDS[(offset+i)%5]
        actions.append(action(proposals[early],proposals[late],theta,trigger,source,name=f'train_c6_{i}'))
    return actions,dict(seed=seed,theta_cycle_offset=offset,late_sources=sources)

def main():
    prereg();assert (HERE/'OPENING_RECORD.receipt.json').exists();assert read(HERE/'ETA_SOURCE_DECISION.json')['switch_requery']
    old=D/'crosstype5_repair_gate_late_eta_v1';c4=D/'crosstype4_no_repair_schedule_diagnosis_v1';base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');rows=[];profiles={};chosen=[]
    with np.load(old/'TRAINING_INPUTS.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
    with np.load(c4/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:oldindices=z['candidate_indices']
    catalogs=read(c4/'CANDIDATES.json')['states'];oldadded=read(old/'TRAINING_ADDED_ACTIONS.json');oldadded={(r['controller'],r['state_index']):r['added_actions'] for r in oldadded};vectors=[];meta={k:[] for k in ('diagonal','contains_zero','pair_index','switch_index','thresholds','trigger_indices')}
    for ni,name in enumerate(base['controller_order']):
        nv=[];nm={k:[] for k in meta}
        for si,s in enumerate(states):
            aa,seed=draws(name,s,inputs['proposals'][si]);prior=[dict(catalogs[si]['actions'][int(j)],late_source='given') for j in oldindices[ni,si]]+[x['action'] for x in oldadded[(name,si)]]
            assert len(prior)==44
            allactions=prior+aa;nv.append([encode(a) for a in allactions]);pairs={};d=[];zero=[];pair=[];switch=[];theta=[];trigger=[]
            for a in allactions:
                early=np.asarray(a['eta_a'],np.float64);late=np.asarray(a['eta_b'],np.float64);rq=a.get('late_source','given')=='requery';key=(early.tobytes(),late.tobytes(),rq);pairs.setdefault(key,len(pairs));d.append(not rq and np.array_equal(early,late));zero.append(not np.any(early) or not np.any(late));pair.append(pairs[key]);switch.append(a.get('progress_threshold',.9681620884830543) if a['switch_mode']=='progress' else a['switch_step']/700);theta.append(a.get('progress_threshold',.9681620884830543 if a['switch_mode']=='progress' else 0.));trigger.append(list(TRIGGERS).index(a.get('trigger','none')))
            for k,v in zip(nm,(d,zero,pair,switch,theta,trigger)):nm[k].append(v)
            links=[]
            for slot,a in enumerate(aa,44):
                for e in (a['eta_a'],a['eta_b']):assert not any(np.array_equal(e,base['eta'][j]) for j in (10,13))
                c,cp=case(base,name,s,a,0,si,slot);c['training_slot']=slot;rows.append(c);profiles[cp['controller_uid']]=cp;links.append(dict(slot=slot,action=a,rollout_uid=c['rollout_uid']))
            chosen.append(dict(controller=name,state_index=si,state_uid=s['state_uid'],seed_receipt=seed,added_actions=links))
        vectors.append(nv)
        for k in meta:meta[k].append(nm[k])
    assert len(rows)==52704
    np.savez_compressed(HERE/'TRAINING_INPUTS.npz',names=inputs['names'],types=inputs['types'],h=inputs['h'],c=inputs['c'],proposals=inputs['proposals'],action=np.asarray(vectors,np.float32),**{k:np.asarray(v) for k,v in meta.items()})
    assert np.array_equal(np.asarray(vectors,np.float32)[:,:,:44,:14],inputs['action'])
    write(HERE/'TRAINING_ADDED_ACTIONS.json',chosen);register('training',dict(base,states=states,phase='training'),rows,profiles)
    missing=read(HERE/'training_MISSING_CASES.json');aliases={}
    with rollout_db.connect(True) as db:
        for i,c in enumerate(missing):
            a=infer(c,base,db)
            if a:aliases[c['rollout_uid']]=a
            if i%1000==0:print('Local trace preflight',i,'/',len(missing),len(aliases),flush=True)
    execution=[c for c in missing if c['rollout_uid'] not in aliases];write(HERE/'training_LOCAL_EQUIVALENCE.json',aliases);write(HERE/'training_EXECUTION_CASES.json',execution)
    write(HERE/'training_BUDGET.json',dict(utc=utc(),requests=len(rows),old44_labels_directly_reused=122*48*44,exact_DB_reuse=len(rows)-len(missing),genuine_DB_missing=len(missing),local_trace_equivalence=len(aliases),actual_execution_cap=len(execution),action_dimension=19,new_slots=9))
    seal('training_EXECUTION_FROZEN.receipt.json',[HERE/n for n in ('training_protocol.json','training_CASES.json','training_PREFLIGHT.receipt.json','TRAINING_INPUTS.npz','TRAINING_ADDED_ACTIONS.json','training_LOCAL_EQUIVALENCE.json','training_EXECUTION_CASES.json','training_BUDGET.json','design_training_c6.py','worker_c6.py','actions_c6.py','execution_c6.py','schedule_c6.py','reuse_c6.py','OPENING_RECORD.receipt.json','EARLY_TERMINATION_EQUIVALENCE.json')],before_rollout=True)
    print('Training execution cap',len(execution),flush=True)

if __name__=='__main__':main()
