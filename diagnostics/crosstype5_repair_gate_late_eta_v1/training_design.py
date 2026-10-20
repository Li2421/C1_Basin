"""Six exploration, four requery actions and only missing safety seed1."""
from common import *
from actions_v3 import action,case
from action_encoding import encode
from registry import register
from local_reuse import infer
from frozen_networks import stable_int,CENTER,RADIUS

def draws(name,s,proposals):
    rng=np.random.default_rng(stable_int('crosstype5-exploration',202610195101,name,s['state_uid']));early=rng.choice(8,3,replace=False);rows=[];coordinates=[]
    for k in range(6):
        if k<3:a=np.asarray(proposals[int(early[k])]);za=None
        else:za=rng.uniform(-3,3,3);a=np.asarray(CENTER,np.float64)+np.asarray(RADIUS,np.float64)*np.tanh(za)
        zb=rng.uniform(-3,3,3);b=np.asarray(CENTER,np.float64)+np.asarray(RADIUS,np.float64)*np.tanh(zb);rows.append(action(a,b,'progress',700,f'exploration_{k}'));coordinates.append(dict(before_tanh_a=za,before_tanh_b=zb,frozen_early_index=int(early[k]) if k<3 else None))
    rr=np.random.default_rng(stable_int('crosstype5-requery-training',202610195102,name,s['state_uid']));chosen=rr.choice(8,2,replace=False)
    for i in chosen:
        for mode in ('fixed','progress'):rows.append(action(proposals[int(i)],proposals[0],mode,500,f'requery_{int(i)}_{mode}','requery'))
    assert len(rows)==10
    return rows,coordinates

def main():
    prereg();assert read(HERE/'REQUERY_EQUIVALENCE_v3.json')['passed'];assert (HERE/'EARLY_TERMINATION_EQUIVALENCE_v3.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');old=D/'crosstype4_no_repair_schedule_diagnosis_v1';rows=[];profiles={};chosen=[];dist=[]
    with np.load(old/'STATE_INPUTS.npz',allow_pickle=False) as z:proposals=z['proposals'];h=z['h'];context=z['c']
    # These were already frozen and evaluated; no old rollout is re-requested.
    with np.load(old/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:old_indices=z['candidate_indices'];assert list(z['names'])==base['controller_order']
    old_catalogues=read(old/'CANDIDATES.json')['states'];all_vectors=[];all_meta={k:[] for k in ('diagonal','contains_zero','pair_index','switch_index')}
    for ni,name in enumerate(base['controller_order']):
        vectors=[];metadata={k:[] for k in all_meta}
        for si,s in enumerate(states):
            aa,coords=draws(name,s,proposals[si]);actions=[dict(old_catalogues[si]['actions'][int(idx)],late_source='given') for idx in old_indices[ni,si]]+aa;assert len(actions)==44;vectors.append([encode(a) for a in actions]);pairs={};diag=[];zeros=[];pairids=[];switchids=[]
            for a in actions:
                early=np.asarray(a['eta_a'],np.float64);late=np.asarray(a['eta_b'],np.float64);rq=a.get('late_source','given')=='requery';pair=(early.tobytes(),late.tobytes(),rq);pairs.setdefault(pair,len(pairs));diag.append(not rq and np.array_equal(early,late));zeros.append(np.all(early==0) or np.all(late==0));pairids.append(pairs[pair]);switchids.append(5 if a['switch_mode']=='progress' else (200,300,400,500,600).index(a['switch_step']))
            for key,v in zip(metadata,(diag,zeros,pairids,switchids)):metadata[key].append(v)
            links=[]
            for slot,a in enumerate(aa,34):
                c,cp=case(base,name,s,a,0,si,slot);c['training_slot']=slot;rows.append(c);profiles[cp['controller_uid']]=cp;links.append(dict(slot=slot,action=a,rollout_uid=c['rollout_uid']))
                if slot<40:
                    for e in (a['eta_a'],a['eta_b']):assert not any(np.array_equal(e,base['eta'][j]) for j in (10,13))
                    dist.append([min(np.linalg.norm(np.asarray(a['eta_b'])-base['eta'][j]) for j in (10,13)),min(np.linalg.norm((np.asarray(a['eta_b'])-base['eta'][j])/np.asarray(RADIUS)) for j in (10,13))])
            c,cp=case(base,name,s,action([0,0,0],[0,0,0],name='safety_seed1'),1,si,483);c['training_slot']=26;rows.append(c);profiles[cp['controller_uid']]=cp;chosen.append(dict(controller=name,state_uid=s['state_uid'],state_index=si,exploration_unconstrained=coords,added_actions=links,safety_seed1_uid=c['rollout_uid']))
        all_vectors.append(vectors)
        for key in metadata:all_meta[key].append(metadata[key])
    assert len(rows)==64416
    register('training',dict(base,states=states,phase='training'),rows,profiles);write(HERE/'TRAINING_ADDED_ACTIONS.json',chosen);np.savez_compressed(HERE/'TRAINING_INPUTS.npz',names=base['controller_order'],types=[base['types'][n] for n in base['controller_order']],h=h,c=context,proposals=proposals,action=np.asarray(all_vectors,np.float32),**{k:np.asarray(v) for k,v in all_meta.items()});write(HERE/'EXPLORATION_ANCHOR_DISTANCES.json',dict(utc=utc(),artificial_anchors_not_inserted=True,late_eta_nearest_distance_quantiles=dict(raw=np.quantile(np.asarray(dist)[:,0],[0,.01,.1,.5,.9,.99,1]),normalized=np.quantile(np.asarray(dist)[:,1],[0,.01,.1,.5,.9,.99,1]))))
    missing=read(HERE/'training_MISSING_CASES.json');aliases={}
    with rollout_db.connect(True) as db:
        for i,c in enumerate(missing):
            a=infer(c,base,db)
            if a:aliases[c['rollout_uid']]=a
            if i%1000==0:print('Preflight local equivalence',i,'/',len(missing),len(aliases),flush=True)
    execution=[c for c in missing if c['rollout_uid'] not in aliases];write(HERE/'training_LOCAL_EQUIVALENCE.json',aliases);write(HERE/'training_EXECUTION_CASES.json',execution);write(HERE/'training_BUDGET.json',dict(utc=utc(),requests=64416,exploration=35136,requery=23424,safety_seed1_requests=5856,old34_reused_directly_from_frozen_bundle=199104,exact_DB_reuse=len(rows)-len(missing),DB_missing=len(missing),early_equivalence=len(aliases),actual_execution_cap=len(execution),input_dimension=14))
    from opening_record import main as opening
    opening()
    deps=['training_protocol.json','training_CASES.json','training_PREFLIGHT.receipt.json','TRAINING_ADDED_ACTIONS.json','TRAINING_INPUTS.npz','EXPLORATION_ANCHOR_DISTANCES.json','training_LOCAL_EQUIVALENCE.json','training_EXECUTION_CASES.json','training_BUDGET.json','training_design.py','action_encoding.py','actions_v3.py','execution_v3.py','requery_generator_v2.py','requery_execution_v3.py','requery_client.py','requery_helper.py','snapshot_conditioning.py','local_reuse.py','worker.py','EARLY_TERMINATION_EQUIVALENCE_v3.receipt.json','REQUERY_EQUIVALENCE_v3.receipt.json','OPENING_RECORD.receipt.json']
    seal('training_EXECUTION_FROZEN.receipt.json',[HERE/n for n in deps],before_submission=True)
    print('Training actual cap',len(execution),flush=True)
if __name__=='__main__':main()
