"""Outcome-free eight proposals, full schedules and fixed sparse training draw."""
from common import *
from frozen_networks import stable_int,Generator,merge_initialized,eta_mean,eta_from_noise,scalar_environment_fields
from legacy_conditioning import ReadOnlyConditioning,physical_signature,normalized
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
from flax import serialization
import jax,jax.numpy as jnp
import csv,copy
jax.config.update('jax_enable_x64',False)
OLD=D/'crosstype1c_two_eta_schedule_v1'
TIMES=(200,300,400,500,600)
def catalog(etas):
    rows=[]
    for i in range(8):
        for j in range(8):
            for k,t in enumerate(TIMES):
                rows.append(dict(candidate_index=len(rows),early_index=i,late_index=j,eta_a=etas[i].tolist(),eta_b=etas[j].tolist(),switch_step=500 if i==j else t,requested_switch_step=t,switch_mode='fixed' if i==j else 'step',name=f'g{i}_fixed' if i==j else f'g{i}_to_g{j}_t{t}',unique_representative=i!=j or t==500))
    return rows
def encode(a):return [*a['eta_a'],*a['eta_b'],*(np.asarray(a['eta_b'])-a['eta_a']),a['switch_step']/700]
def draw(name,state_uid):
    rng=np.random.default_rng(stable_int('crosstype1c-training-actions',202610190101,name,state_uid));diagonal=[(i*8+i)*5+3 for i in range(8)];picked=[]
    for k in range(5):picked.extend(rng.choice([p*5+k for p in range(64) if p//8!=p%8],3,replace=False).tolist())
    pool=[p*5+k for p in range(64) if p//8!=p%8 for k in range(5) if p*5+k not in picked]
    picked.extend(rng.choice(pool,3,replace=False).tolist());result=diagonal+picked
    assert len(set(result))==26 and all(sum(a%5==k for a in picked)>=3 for k in range(5))
    return result
def fresh_states():
    existing=set();uids=set();contents=set();unknown={};count=0
    with rollout_db.connect(True) as db:
        for r in db.execute('SELECT state_uid,content_hash,physical_state_json,goals_geometry_json,identity_quality FROM state'):
            count+=1;uids.add(r['state_uid']);contents.add(r['content_hash']);physical=json.loads(r['physical_state_json']) if r['physical_state_json'] else {}
            if isinstance(physical,dict) and 'goals' not in physical:
                g=json.loads(r['goals_geometry_json']) if r['goals_geometry_json'] else {}
                if isinstance(g,dict) and 'goals' in g:physical={**physical,'goals':g['goals']}
            signature=physical_signature(physical)
            if signature is None:unknown[r['identity_quality']]=unknown.get(r['identity_quality'],0)+1
            else:existing.add(signature)
    hashes_before=len(existing);native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');states=[];rejected=[];seed=202610193001
    while len(states)<48:
        raw=generate('crosstype3_new_evaluation',1,seed,native)[0];s=normalized(raw);signature=physical_signature(s['physical']);assert signature is not None
        if signature in existing or s['state_uid'] in uids or s['content_hash'] in contents:rejected.append(seed)
        else:s['study_index']=len(states);states.append(s);existing.add(signature)
        seed+=1;assert seed<202610199001
    write(HERE/'NEW_STATE_DEDUP.json',dict(utc=utc(),database_rows=count,physical_hashes_recoverable=hashes_before,unrecoverable_by_quality=unknown,all_UID_content_checked=True,includes_registered_crosstype1b_and2_states=True,first_seed=202610193001,last_seed=seed-1,rejected_seeds=rejected,new_states=48,outcome_blind=True))
    return states
def main():
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed'] and read(HERE/'GENERATOR_REPRODUCTION.json')['passed'];assert not (HERE/'CANDIDATES.receipt.json').exists()
    base=read(HERE/'BASE_PROTOCOL.json');train=read(OLD/'TRAINING_STATES.json');fresh=fresh_states();states=train+fresh
    assert not {physical_signature(s['physical']) for s in train}&{physical_signature(s['physical']) for s in fresh}
    base=read(HERE/'BASE_PROTOCOL.json');names=base['controller_order'];types=[base['types'][n] for n in names]
    assert len(names)==104 and len(set(types))==16
    norm=read(D/'orthoflow3_generator_critic_v1/normalization.json');n=norm['scenarios']['ring_exchange'];cdim=len(norm['environment_keys'])+3;gm=Generator();parameters=serialization.from_bytes(merge_initialized(gm,{k:len(v['h_mean']) for k,v in norm['scenarios'].items()},cdim),(D/'orthoflow3_generator_critic_v1/generator/seed41/checkpoint.msgpack').read_bytes());reference=ReadOnlyConditioning()
    rows=[];hh=[];cc=[];actions=[];etas_all=[];training_indices=[]
    old_candidates=read(OLD/'CANDIDATES.json')['states']
    for si,s in enumerate(states):
        h,descriptor=reference.conditioning(dict(uid=s['state_uid'],physical=s['physical']));c=np.zeros(cdim,np.float32);values=scalar_environment_fields(descriptor)
        for j,k in enumerate(norm['environment_keys']):c[j]=values.get(k,0.)
        c[-1]=1.;hm=(np.asarray(h,np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32);cm=(c-np.asarray(n['c_mean'],np.float32))/np.asarray(n['c_std'],np.float32)
        raw=np.asarray(gm.apply(parameters,jnp.asarray(hm[None]),jnp.asarray(cm[None]),method=gm.ring))[0];seed=stable_int('generator-v1-proposals',41,'ring_exchange',s['state_uid']);noise=np.random.default_rng(seed).standard_normal((7,3)).astype(np.float32)
        proposals=np.asarray([np.asarray(eta_mean(jnp.asarray(raw)),float),*np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(7,0),jnp.asarray(noise)),float)])
        assert not any(np.array_equal(e,base['eta'][anchor]) for e in proposals for anchor in (10,13))
        if si<48:assert proposals.tobytes()==np.asarray(old_candidates[si]['proposals'],np.float64).tobytes()
        aa=catalog(proposals);rows.append(dict(state_uid=s['state_uid'],state_index=si,split='training' if si<48 else 'new_evaluation',proposal_seed=seed,proposals=proposals,actions=aa));hh.append(h);cc.append(c);actions.append([encode(a) for a in aa]);etas_all.append(proposals)
    indices=np.asarray([[draw(name,s['state_uid']) for s in train] for name in names],np.int32)
    with np.load(OLD/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:
        for ni,name in enumerate(z['names']):assert np.array_equal(indices[names.index(str(name))],z['candidate_indices'][ni])
    np.savez_compressed(HERE/'TRAINING_ACTIONS.npz',names=names,types=types,state_uids=[s['state_uid'] for s in train],candidate_indices=indices)
    np.savez_compressed(HERE/'STATE_INPUTS.npz',h=np.asarray(hh,np.float32),c=np.asarray(cc,np.float32),action=np.asarray(actions,np.float32),proposals=np.asarray(etas_all,np.float64),state_uids=[s['state_uid'] for s in states])
    write(HERE/'TRAINING_STATES.json',train);write(HERE/'NEW_EVAL_STATES.json',fresh);write(HERE/'BASE_PROTOCOL.json',base);write(HERE/'CANDIDATES.json',dict(utc=utc(),states=rows,controller_independent=True,total_indices_per_state=320,unique_actions_per_state=288))
    seal('CANDIDATES.receipt.json',[HERE/f for f in ('CANDIDATES.json','STATE_INPUTS.npz','TRAINING_ACTIONS.npz','TRAINING_STATES.json','NEW_EVAL_STATES.json','NEW_STATE_DEDUP.json','BASE_PROTOCOL.json','candidates.py')],before_new_state_rollouts=True,eta13_eta10_excluded_from_candidates=True)
    print('Sealed eight proposals, 320 indices/288 unique actions and 26 sparse labels for each training pair',flush=True)
if __name__=='__main__':main()
