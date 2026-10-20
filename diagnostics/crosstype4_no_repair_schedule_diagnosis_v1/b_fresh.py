"""Outcome-blind 24 fresh states and frozen-generator proposals; no rollout."""
from common import *
from legacy_conditioning import physical_signature,normalized,ReadOnlyConditioning
from frozen_networks import stable_int,Generator,merge_initialized,eta_mean,eta_from_noise,scalar_environment_fields
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
from b_candidates import catalog,encode
from flax import serialization
import jax,jax.numpy as jnp

def fresh_states():
    existing=set();uids=set();contents=set();unknown={};count=0
    with rollout_db.connect(True) as db:
        for r in db.execute('SELECT state_uid,content_hash,physical_state_json,goals_geometry_json,identity_quality FROM state'):
            count+=1;uids.add(r['state_uid']);contents.add(r['content_hash']);p=json.loads(r['physical_state_json']) if r['physical_state_json'] else {}
            if isinstance(p,dict) and 'goals' not in p:
                g=json.loads(r['goals_geometry_json']) if r['goals_geometry_json'] else {}
                if isinstance(g,dict) and 'goals' in g:p={**p,'goals':g['goals']}
            sig=physical_signature(p)
            if sig is None:unknown[r['identity_quality']]=unknown.get(r['identity_quality'],0)+1
            else:existing.add(sig)
    original=len(existing);native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');states=[];rejected=[];seed=202610194301
    while len(states)<24:
        raw=generate('crosstype4_no_repair_new_evaluation',1,seed,native)[0];s=normalized(raw);sig=physical_signature(s['physical']);assert sig is not None
        if sig in existing or s['state_uid'] in uids or s['content_hash'] in contents:rejected.append(seed)
        else:s['study_index']=len(states);states.append(s);existing.add(sig)
        seed+=1;assert seed<202610195301
    write(HERE/'NEW_STATE_DEDUP.json',dict(utc=utc(),database_rows=count,recoverable_physical_hashes=original,unrecoverable_by_quality=unknown,all_UID_and_content_checked=True,draw_seed_start=202610194301,last_seed=seed-1,rejected_seeds=rejected,new_states=24,namespace='crosstype4_no_repair_new_evaluation',no_rollouts=True))
    return states

def main():
    prereg();assert not (HERE/'NEW_EVAL_INPUTS.receipt.json').exists();states=fresh_states();norm=read(D/'orthoflow3_generator_critic_v1/normalization.json');normal=norm['scenarios']['ring_exchange'];cdim=len(norm['environment_keys'])+3;gm=Generator();checkpoint=D/'orthoflow3_generator_critic_v1/generator/seed41/checkpoint.msgpack';parameters=serialization.from_bytes(merge_initialized(gm,{k:len(v['h_mean']) for k,v in norm['scenarios'].items()},cdim),checkpoint.read_bytes());reference=ReadOnlyConditioning();rows=[];hh=[];cc=[];vectors=[];etas=[]
    for si,s in enumerate(states):
        h,descriptor=reference.conditioning(dict(uid=s['state_uid'],physical=s['physical']));c=np.zeros(cdim,np.float32);values=scalar_environment_fields(descriptor)
        for j,k in enumerate(norm['environment_keys']):c[j]=values.get(k,0.)
        c[-1]=1.;hm=(np.asarray(h,np.float32)-np.asarray(normal['h_mean'],np.float32))/np.asarray(normal['h_std'],np.float32);cm=(c-np.asarray(normal['c_mean'],np.float32))/np.asarray(normal['c_std'],np.float32);raw=np.asarray(gm.apply(parameters,jnp.asarray(hm[None]),jnp.asarray(cm[None]),method=gm.ring))[0];seed=stable_int('generator-v1-proposals',41,'ring_exchange',s['state_uid']);noise=np.random.default_rng(seed).standard_normal((7,3)).astype(np.float32);proposals=np.asarray([np.asarray(eta_mean(jnp.asarray(raw)),float),*np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(7,0),jnp.asarray(noise)),float)]);aa=catalog(proposals)
        rows.append(dict(state_uid=s['state_uid'],state_index=si,proposal_seed=seed,proposals=proposals,actions=aa));hh.append(h);cc.append(c);vectors.append([encode(a) for a in aa]);etas.append(proposals)
    write(HERE/'NEW_EVAL_STATES.json',states);write(HERE/'NEW_CANDIDATES.json',dict(utc=utc(),states=rows,generator_checkpoint_sha256=sha(checkpoint),proposals_independent_of_controller=True))
    np.savez_compressed(HERE/'NEW_EVAL_INPUTS.npz',h=np.asarray(hh,np.float32),c=np.asarray(cc,np.float32),action=np.asarray(vectors,np.float32),proposals=np.asarray(etas,np.float64),state_uids=[s['state_uid'] for s in states])
    seal('NEW_EVAL_INPUTS.receipt.json',[HERE/n for n in ('NEW_EVAL_STATES.json','NEW_STATE_DEDUP.json','NEW_CANDIDATES.json','NEW_EVAL_INPUTS.npz','b_fresh.py','b_candidates.py','legacy_conditioning.py','frozen_networks.py')],before_any_fresh_closed_loop=True,source_generator_checkpoint_sha256=sha(checkpoint))
if __name__=='__main__':main()
