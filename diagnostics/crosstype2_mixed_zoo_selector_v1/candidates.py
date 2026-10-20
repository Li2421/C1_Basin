"""Recompute frozen proposals; prepare outcome-blind 16-action state packets."""
from common import *
from generator_reference import ReadOnlyConditioning,physical_signature,normalized,action
from frozen_networks import Generator,Critic,merge_initialized,eta_mean,eta_from_noise,stable_int,CENTER,RADIUS,scalar_environment_fields
from flax import serialization
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',False)

OLD=D/'crosstype1b_generator_critic_pipeline_v1'
LEARN=D/'orthoflow3_generator_critic_v1'


def networks():
    norm=read(LEARN/'normalization.json');n=norm['scenarios']['ring_exchange']
    dims={sc:len(v['h_mean']) for sc,v in norm['scenarios'].items()};cdim=len(norm['environment_keys'])+3
    gm,cm=Generator(),Critic()
    gp=serialization.from_bytes(merge_initialized(gm,dims,cdim),(LEARN/'generator/seed41/checkpoint.msgpack').read_bytes())
    cp=serialization.from_bytes(merge_initialized(cm,dims,cdim,critic=True),(LEARN/'critic/seed23/checkpoint.msgpack').read_bytes())
    return norm,n,gm,cm,gp,cp


def infer(h,c,uid,models):
    norm,n,gm,cm,gp,cp=models
    hh=(np.asarray(h,np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32)
    cc=(np.asarray(c,np.float32)-np.asarray(n['c_mean'],np.float32))/np.asarray(n['c_std'],np.float32)
    raw=np.asarray(gm.apply(gp,jnp.asarray(hh[None]),jnp.asarray(cc[None]),method=gm.ring))[0]
    seed=stable_int('generator-v1-proposals',41,'ring_exchange',uid);rng=np.random.default_rng(seed)
    mean=np.asarray(eta_mean(jnp.asarray(raw)),float)
    samples=np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(4,0),jnp.asarray(rng.standard_normal((4,3)),jnp.float32)),float)
    proposals=np.asarray([mean,*samples])
    logits=np.asarray(cm.apply(cp,jnp.asarray(np.repeat(hh[None],5,0)),jnp.asarray(np.repeat(cc[None],5,0)),jnp.asarray((proposals-CENTER)/RADIUS),method=cm.ring))
    return proposals,1/(1+np.exp(-logits)),seed


def reproduce():
    prereg();models=networks();previous=read(OLD/'CANDIDATES.json')['states'];checks=[]
    with np.load(OLD/'STATE_INPUTS.npz',allow_pickle=False) as z:
        for i,row in enumerate(previous):
            proposals,scores,seed=infer(z['h'][i],z['c'][i],row['state_uid'],models)
            a=np.asarray(row['proposals']);b=np.asarray(row['original_critic_scores'])
            checks.append(dict(state_uid=row['state_uid'],proposals_bitwise=proposals.tobytes()==a.tobytes(),critic_scores_bitwise=scores.astype(float).tobytes()==b.tobytes(),proposal_max_error=float(np.max(np.abs(proposals-a)))))
    passed=all(c['proposals_bitwise'] and c['critic_scores_bitwise'] for c in checks)
    write(HERE/'FROZEN_COMPONENT_REPRODUCTION.json',dict(utc=utc(),passed=passed,count=len(checks),checks=checks,x64=False,backend=jax.default_backend(),physical_rollouts=0))
    assert passed
    print('Frozen generator and original critic bitwise match all 72 historical states')


def fresh_states():
    native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    hashes=set();ids=set();contents=set();unresolved={};scanned=0
    with rollout_db.connect(True) as db:
        for r in db.execute('SELECT state_uid,content_hash,physical_state_json,goals_geometry_json,identity_quality FROM state'):
            scanned+=1;ids.add(r['state_uid']);contents.add(r['content_hash'])
            value=json.loads(r['physical_state_json']) if r['physical_state_json'] else {}
            if isinstance(value,dict) and 'goals' not in value:
                geom=json.loads(r['goals_geometry_json']) if r['goals_geometry_json'] else {}
                if isinstance(geom,dict) and 'goals' in geom:value={**value,'goals':geom['goals']}
            signature=physical_signature(value)
            if signature is None:unresolved[str(r['identity_quality'])]=unresolved.get(str(r['identity_quality']),0)+1
            else:hashes.add(signature)
    rows=[];rejected=[];seed=202610182001;count=len(hashes)
    while len(rows)<24:
        raw=generate('crosstype2_new_evaluation',1,seed,native)[0];row=normalized(raw);signature=physical_signature(row['physical'])
        assert signature is not None
        if signature in hashes or row['state_uid'] in ids or row['content_hash'] in contents:rejected.append(seed)
        else:row['study_index']=len(rows);rows.append(row);hashes.add(signature)
        seed+=1;assert seed<202610183001
    write(HERE/'NEW_EVAL_STATES.json',rows)
    write(HERE/'NEW_STATE_DEDUP.json',dict(utc=utc(),DB_rows_scanned=scanned,unique_recoverable_physical_hashes=count,unresolved_physical_by_quality=unresolved,all_UIDs_and_content_hashes_scanned=True,draw_seed_start=202610182001,last_draw_seed=seed-1,rejected_duplicate_seeds=rejected,new_states=24,outcome_blind=True,legacy_unrecoverable_coordinates_remain_gap=True))
    return rows


def design():
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed'];assert read(HERE/'FROZEN_COMPONENT_REPRODUCTION.json')['passed']
    assert not (HERE/'CANDIDATES.receipt.json').exists()
    train=read(OLD/'TRAINING_STATES.json');fresh=fresh_states();models=networks();norm=models[0];reference=ReadOnlyConditioning()
    assert len(train)==48 and not {physical_signature(s['physical']) for s in train}&{physical_signature(s['physical']) for s in fresh}
    write(HERE/'TRAINING_STATES.json',train);old=read(OLD/'CANDIDATES.json');eta=read(OLD/'BASE_PROTOCOL.json')['eta']
    shortlist=read(HERE/'RHR_SHORTLIST.json');rules=[shortlist[k] for k in ('fixed','RHR_b','RHR_c')]
    names=list(old['action_names'])+['RHR','RHR_b','RHR_c'];records=[];hs=[];cs=[];vectors=[]
    with np.load(OLD/'STATE_INPUTS.npz',allow_pickle=False) as z:oldh=z['h'][:48].copy();oldc=z['c'][:48].copy()
    for i,s in enumerate(train+fresh):
        if i<48:h,c=oldh[i],oldc[i];row=dict(old['states'][i]);actions=[dict(a) for a in row['actions']]
        else:
            h,descriptor=reference.conditioning(dict(uid=s['state_uid'],physical=s['physical']))
            c=np.zeros(len(norm['environment_keys'])+3,np.float32);values=scalar_environment_fields(descriptor)
            for j,key in enumerate(norm['environment_keys']):c[j]=values.get(key,0.)
            c[-1]=1.;proposals,scores,seed=infer(h,c,s['state_uid'],models);actions=[]
            for proposal in proposals:actions.extend([action(proposal,proposal,'fixed'),action(proposal,eta[10],'progress')])
            actions.extend([action([0,0,0],[0,0,0],'fixed',True),action(eta[13],eta[10],'step500'),action(eta[13],eta[10],'progress')])
            actions=[dict(name=name,**a) for name,a in zip(names,actions)]
            row=dict(state_uid=s['state_uid'],state_index=i,split='new_evaluation',proposal_seed=seed,proposals=proposals,original_critic_scores=scores,original_pipeline_choice=int(np.argmax(scores))*2,generator_only_choice=1)
        actions.extend(dict(name=name,**action([0,0,0],[0,0,0],'fixed'),rhr_rule=rule) for name,rule in zip(names[13:],rules))
        encoded=[]
        for a in actions:
            rule=a.get('rhr_rule');onehot=[float(a['switch_mode']==m) for m in ('fixed','progress','step500')]
            oldvector=[*((np.asarray(a['eta_a'])-CENTER)/RADIUS),*((np.asarray(a['eta_b'])-CENTER)/RADIUS),*onehot,float(a['safety_baseline'])]
            encoded.append([*oldvector,float(rule is not None),*[float(rule is not None and rule['mode']==m) for m in ('always','slow','progress')],rule['steps']/40 if rule else 0.,rule['speed']/.35 if rule else 0.])
        row.update(actions=actions,state_index=i,split='training' if i<48 else 'new_evaluation');records.append(row);vectors.append(encoded);hs.append(h);cs.append(c)
    np.savez_compressed(HERE/'STATE_INPUTS.npz',h=np.asarray(hs,np.float32),c=np.asarray(cs,np.float32),action=np.asarray(vectors,np.float32),state_uids=[s['state_uid'] for s in train+fresh],split=['training']*48+['new_evaluation']*24)
    write(HERE/'CANDIDATES.json',dict(utc=utc(),action_names=names,states=records,proposals_controller_independent=True,old_training_proposals_reused_exactly=True))
    seal('CANDIDATES.receipt.json',[HERE/n for n in ('CANDIDATES.json','STATE_INPUTS.npz','TRAINING_STATES.json','NEW_EVAL_STATES.json','NEW_STATE_DEDUP.json','RHR_SHORTLIST.json','candidates.py','generator_reference.py','frozen_networks.py')],before_new_evaluation_rollout=True,new_physical_rollouts=0)
    print('Prepared 48 old training and 24 fresh states, 16 actions, no evaluation rollouts')


if __name__=='__main__':globals()[sys.argv[1]]()
