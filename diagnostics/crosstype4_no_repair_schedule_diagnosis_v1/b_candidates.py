"""Frozen proposals plus zero, canonical IDs and prespecified 34-label draw."""
from common import *
from actions import action,TIMES
from frozen_networks import stable_int
MODES=(*TIMES,'progress')

def catalog(etas):
    etas=[*np.asarray(etas,np.float64).tolist(),[0.,0.,0.]];rows=[]
    for i in range(9):
        for j in range(9):
            for k,t in enumerate(MODES):
                a=action(etas[i],etas[j],'progress' if t=='progress' else 'fixed',700 if t=='progress' else t)
                rows.append(dict(a,candidate_index=i*54+j*6+k,early_index=i,late_index=j,requested_switch=t,unique_representative=i!=j or k==3))
    assert sum(r['unique_representative'] for r in rows)==441
    return rows

def encode(a):
    z=lambda e:float(np.all(np.asarray(e)==0))
    return [*a['eta_a'],*a['eta_b'],*(np.asarray(a['eta_b'])-a['eta_a']),0. if a['switch_mode']=='progress' else a['switch_step']/700,z(a['eta_a']),z(a['eta_b']),float(a['switch_mode']=='progress')]

def legacy_draw(name,state_uid):
    rng=np.random.default_rng(stable_int('crosstype1c-training-actions',202610190101,name,state_uid));diagonal=[(i*8+i)*5+3 for i in range(8)];picked=[]
    for k in range(5):picked.extend(rng.choice([p*5+k for p in range(64) if p//8!=p%8],3,replace=False).tolist())
    pool=[p*5+k for p in range(64) if p//8!=p%8 for k in range(5) if p*5+k not in picked];picked.extend(rng.choice(pool,3,replace=False).tolist())
    return diagonal+picked

def remap(index):return (index//40)*54+(index//5%8)*6+index%5

def main():
    prereg();assert (HERE/'EXECUTION_EQUIVALENCE.receipt.json').exists();base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');old=D/'crosstype3_unseen_controller_schedule_v1';oldc=read(old/'CANDIDATES.json')['states'][:48];names=base['controller_order'];indices=[];old_indices=[];balance={};rank={s['state_uid']:k for k,s in enumerate(sorted(states,key=lambda s:s['state_uid']))}
    for name in names:
        per=[];legacy=[];time_rng=np.random.default_rng(stable_int('crosstype4-extra-training',202610194201,name,'time_cycle'));cycle=time_rng.permutation(5);counts=np.zeros(5,int)
        for s in states:
            legacy_ids=legacy_draw(name,s['state_uid']);old26=[remap(i) for i in legacy_ids];rng=np.random.default_rng(stable_int('crosstype4-extra-training',202610194201,name,s['state_uid']));late=rng.choice(8,2,replace=False);early=rng.choice(8,2,replace=False);ks=[int(cycle[(rank[s['state_uid']]*4+k)%5]) for k in range(4)]
            for k in ks:counts[k]+=1
            extra=[483,*[8*54+int(j)*6+ks[k] for k,j in enumerate(late)],*[int(i)*54+8*6+ks[2+k] for k,i in enumerate(early)]];pairs=rng.choice([i*9+j for i in range(8) for j in range(8) if i!=j],3,replace=False);extra.extend([int(p)//9*54+int(p)%9*6+5 for p in pairs]);assert len(extra)==8 and len(set(old26+extra))==34;per.append(old26+extra);legacy.append(legacy_ids)
        assert counts.max()-counts.min()<=1;balance[name]={str(t):int(n) for t,n in zip(TIMES,counts)};indices.append(per);old_indices.append(legacy)
    indices=np.asarray(indices,np.int32);old_indices=np.asarray(old_indices,np.int32)
    with np.load(old/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:
        for i,n in enumerate(z['names']):assert np.array_equal(old_indices[names.index(str(n))],z['candidate_indices'][i])
    catalogues=[dict(state_uid=s['state_uid'],state_index=i,proposals=c['proposals'],actions=catalog(c['proposals'])) for i,(s,c) in enumerate(zip(states,oldc))]
    with np.load(old/'STATE_INPUTS.npz',allow_pickle=False) as z:h=z['h'][:48].copy();c=z['c'][:48].copy();proposals=z['proposals'][:48].copy()
    np.savez_compressed(HERE/'STATE_INPUTS.npz',h=h,c=c,proposals=proposals,action=np.asarray([[encode(a) for a in r['actions']] for r in catalogues],np.float32),state_uids=[s['state_uid'] for s in states])
    np.savez_compressed(HERE/'TRAINING_ACTIONS.npz',names=names,types=[base['types'][n] for n in names],candidate_indices=indices,legacy_indices=old_indices,state_uids=[s['state_uid'] for s in states]);write(HERE/'CANDIDATES.json',dict(utc=utc(),states=catalogues,unique=441,catalogue_indices=486,zero_index=8))
    with np.load(old/'CONTROLLER_FEATURES.npz',allow_pickle=False) as z:
        order=[list(z['names']).index(n) for n in names];features=z['features'][order];assert features.shape==(122,24) and np.sum(~np.isfinite(features).all(-1))==10
        np.savez_compressed(HERE/'CONTROLLER_FEATURES.npz',names=names,types=[base['types'][n] for n in names],features=features,feature_names=z['feature_names'])
    write(HERE/'B_CANDIDATE_AUDIT.json',dict(utc=utc(),controllers=122,states=48,labels_per_state=34,requests=199104,old104_requests=129792,raw_increment=69312,extra_time_counts=balance,legacy26draw_bitwise_matches=True,features_reused=True,missing_feature_vectors=10,features_source_sha256=sha(old/'CONTROLLER_FEATURES.npz'),generator_retrained=False,proposals_state_inputs_reused=True))
    seal('CANDIDATES.receipt.json',[HERE/n for n in ('CANDIDATES.json','STATE_INPUTS.npz','TRAINING_ACTIONS.npz','CONTROLLER_FEATURES.npz','B_CANDIDATE_AUDIT.json','b_candidates.py','TRAINING_STATES.json')],before_B_labels=True)
if __name__=='__main__':main()
