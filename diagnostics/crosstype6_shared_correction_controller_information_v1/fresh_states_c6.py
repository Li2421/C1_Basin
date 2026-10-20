"""Outcome-blind round10 generation with complete DB identity/physical dedup."""
from common import *
from legacy_conditioning import normalized,physical_signature
from requery_generator_v2 import FrozenMean
from frozen_networks import stable_int,eta_from_noise
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
import jax.numpy as jnp

def main():
    prereg();assert not (HERE/'NEW_EVAL_INPUTS.receipt.json').exists()
    uids=set();contents=set();physical=set();unknown={};count=0
    with rollout_db.connect(True) as db:
        for row in db.execute('SELECT state_uid,content_hash,physical_state_json,goals_geometry_json,identity_quality FROM state'):
            count+=1;uids.add(row['state_uid']);contents.add(row['content_hash']);p=json.loads(row['physical_state_json']) if row['physical_state_json'] else {}
            if isinstance(p,dict) and 'goals' not in p:
                g=json.loads(row['goals_geometry_json']) if row['goals_geometry_json'] else {}
                if isinstance(g,dict) and 'goals' in g:p=dict(p,goals=g['goals'])
            sig=physical_signature(p)
            if sig is None:unknown[row['identity_quality']]=unknown.get(row['identity_quality'],0)+1
            else:physical.add(sig)
    original=len(physical);native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json');states=[];rejected=[];seed=2026102062
    while len(states)<24:
        s=normalized(generate('crosstype6_shared_correction_new_evaluation',1,seed,native)[0]);sig=physical_signature(s['physical']);assert sig is not None
        if sig in physical or s['state_uid'] in uids or s['content_hash'] in contents:rejected.append(seed)
        else:s['study_index']=len(states);states.append(s);physical.add(sig)
        seed+=1;assert seed<2026103062
    ref=FrozenMean();h=[];context=[];etas=[];checks=[]
    for s in states:
        q=ref.query(s['state_uid'],s['physical']);h.append(q['h']);context.append(q['c'])
        hh=(q['h']-np.asarray(ref.n['h_mean'],np.float32))/np.asarray(ref.n['h_std'],np.float32);cc=(q['c']-np.asarray(ref.n['c_mean'],np.float32))/np.asarray(ref.n['c_std'],np.float32)
        raw=ref.net.apply(ref.parameters,jnp.asarray(hh[None]),jnp.asarray(cc[None]),method=ref.net.ring)
        rng=np.random.default_rng(stable_int('generator-v1-proposals',41,'ring_exchange',s['state_uid']));noise=rng.standard_normal((7,3)).astype(np.float32)
        proposals=np.asarray([q['eta_b'],*np.asarray(eta_from_noise(jnp.repeat(raw,7,0),jnp.asarray(noise)),np.float64)])
        etas.append(proposals);checks.append(dict(state_uid=s['state_uid'],initial_mean=q['eta_b']))
    write(HERE/'NEW_EVAL_STATES.json',states)
    write(HERE/'NEW_STATE_DEDUP.json',dict(utc=utc(),database_rows=count,recoverable_physical_hashes=original,unrecoverable_by_quality=unknown,all_UID_and_content_checked=True,draw_seed_start=2026102062,last_seed=seed-1,rejected_seeds=rejected,new_states=24,namespace='crosstype6_shared_correction_new_evaluation',no_rollouts=True))
    np.savez_compressed(HERE/'NEW_EVAL_INPUTS.npz',h=np.asarray(h,np.float32),c=np.asarray(context,np.float32),proposals=np.asarray(etas,np.float64),state_uids=[s['state_uid'] for s in states])
    seal('NEW_EVAL_INPUTS.receipt.json',[HERE/n for n in ('NEW_EVAL_STATES.json','NEW_STATE_DEDUP.json','NEW_EVAL_INPUTS.npz','fresh_states_c6.py','requery_generator_v2.py','snapshot_conditioning.py','frozen_networks.py')],before_any_fresh_closed_loop=True)
if __name__=='__main__':main()
