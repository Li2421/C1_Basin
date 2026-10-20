"""Outcome-free conditioning and frozen per-state generator actions."""
from common import *
from frozen_networks import Generator,Critic,merge_initialized,eta_mean,eta_from_noise,stable_int,CENTER,RADIUS,scalar_environment_fields
from flax import serialization
from new_benchmark_common.safety_eta3 import ScenarioRuntime,SCENARIOS,state_token
from new_benchmark_common.basin_dataset_v1 import CONDITIONING_FLOW_ROOT,_environment_descriptor
from new_benchmark_common.macflow import load_checkpoint
from ring_exchange.environment import LocalFrameConfig,RingExchangeEnv
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
import jax,jax.numpy as jnp
import copy,csv
jax.config.update('jax_enable_x64',False)

OLD=D/'crosstype1_learned_zoo_selector_v1'
LEARN=D/'orthoflow3_generator_critic_v1'
ACTION_NAMES=tuple(f'g{k}_{kind}' for k in range(5) for kind in ('fixed','progress'))+('safety_baseline','S500','Sprog')


class ReadOnlyConditioning(ScenarioRuntime):
    """Inherit original physical observation/Flow methods, without DB init."""
    def __init__(self):
        self.name='ring_exchange';self.kind='ring';spec=SCENARIOS[self.name]
        self.manifest=read(spec['dataset']/'manifest.json')
        self.config_dict=self.manifest['scenario_config'];self.config=LocalFrameConfig(**self.config_dict)
        self.agent,self.checkpoint_metadata=load_checkpoint(spec['checkpoint'],expected_environment_fingerprint=self.manifest['environment_fingerprint'])
        self.make_env=lambda:RingExchangeEnv(self.config)

    def conditioning(self,state):
        # Exact FrozenNewRuntime.conditioning body; no controller argument.
        env=self.make_env();self.reset(env,state);obs=np.asarray(self.observation(env),float)
        key=jax.random.fold_in(jax.random.PRNGKey(CONDITIONING_FLOW_ROOT),state_token(state['uid']))
        flow=self.flow_world(env,key)
        assert env.step_count==0
        return np.concatenate((obs.reshape(-1),flow.reshape(-1))).tolist(),_environment_descriptor(self)


def physical_signature(value):
    if not isinstance(value,dict):return None
    coordinates=[]
    for names in (('positions','initial_positions'),('velocities','initial_velocities'),('goals',)):
        data=next((value[k] for k in names if k in value),None)
        if data is None:return None
        try:a=np.asarray(data,np.float64).copy()
        except (TypeError,ValueError):return None
        if a.ndim!=2 or a.shape[-1]!=2 or not np.isfinite(a).all():return None
        a[a==0]=0.;coordinates.append((a.shape,a.tobytes().hex()))
    return hashlib.sha256(json.dumps(dict(values=coordinates,timestep=float(value.get('timestep',value.get('step',0)))),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def normalized(s):
    from shared_rollout_db.src.rollout_db import uid
    old=copy.deepcopy(s);physical={k:old['physical'][k] for k in ('positions','velocities','goals')}
    old['original_state_uid']=s['state_uid'];old['original_physical']=s['physical'];old['transport_physical']=physical
    identity=dict(physical=physical,timestep=0,future_root=2026100403,key_token=int(hashlib.sha256(s['uid'].encode()).hexdigest()[:8],16))
    old['content_hash']=uid('content',identity)
    old['state_uid']=uid('state',dict(scenario=s['scenario_uid'],content=old['content_hash']))
    return old


def fresh_states():
    native=read(D/'state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    hashes=set();ids=set();contents=set();unresolved={};scanned=0
    with rollout_db.connect(True) as db:
        for r in db.execute('SELECT state_uid,content_hash,physical_state_json,goals_geometry_json,identity_quality FROM state'):
            scanned+=1;ids.add(r['state_uid']);contents.add(r['content_hash'])
            value=json.loads(r['physical_state_json']) if r['physical_state_json'] else {}
            if isinstance(value,dict) and 'goals' not in value:
                geometry=json.loads(r['goals_geometry_json']) if r['goals_geometry_json'] else {}
                if isinstance(geometry,dict) and 'goals' in geometry:value={**value,'goals':geometry['goals']}
            signature=physical_signature(value)
            if signature is None:unresolved[r['identity_quality']]=unresolved.get(r['identity_quality'],0)+1
            else:hashes.add(signature)
    rows=[];rejected=[];seed=202610180801
    while len(rows)<24:
        raw=generate('crosstype1b_new_evaluation',1,seed,native)[0];row=normalized(raw);signature=physical_signature(row['physical'])
        assert signature is not None
        if signature in hashes or row['state_uid'] in ids or row['content_hash'] in contents:rejected.append(seed)
        else:
            row['study_index']=len(rows);rows.append(row);hashes.add(signature)
        seed+=1
        assert seed<202610181801
    write(HERE/'NEW_EVAL_STATES.json',rows)
    write(HERE/'NEW_STATE_DEDUP.json',dict(utc=utc(),DB_rows_scanned=scanned,unique_recoverable_physical_hashes=len(hashes)-24,
        unresolved_physical_by_quality=unresolved,all_UIDs_and_content_hashes_scanned=True,draw_seed_start=202610180801,
        last_draw_seed=seed-1,rejected_duplicate_seeds=rejected,new_states=24,selection_outcome_blind=True,
        limitation='Legacy states lacking recoverable physical coordinates remain unresolved; no absolute physical-nonoverlap claim for these rows.'))
    return rows


def action(early,late,mode,safety=False):
    return dict(eta_a=list(map(float,early)),eta_b=list(map(float,late)),switch_mode=mode,safety_baseline=bool(safety))


def main():
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed'] and read(HERE/'FROZEN_COMPONENT_REPRODUCTION.json')['passed']
    assert not (HERE/'CANDIDATES.receipt.json').exists()
    pe=read(OLD/'development_evaluation_protocol.json');pt=read(OLD/'development_tuning_protocol.json');pl=read(OLD/'locked_evaluation_protocol.json')
    train=pt['states']+pe['states'];assert len(train)==48 and len({physical_signature(s['physical']) for s in train})==48
    fresh=fresh_states();assert not {physical_signature(s['physical']) for s in train}&{physical_signature(s['physical']) for s in fresh}
    write(HERE/'TRAINING_STATES.json',train)
    with (OLD/'ZOO_MANIFEST.csv').open() as f:zoo=[r for r in csv.DictReader(f) if r['type_code']!='implicit']
    names=[r['controller'] for r in zoo];types=[r['type_code'] for r in zoo];profiles={}
    for name in names:profiles[name]=(pl if name in pl['controller_order'] else pe)['profiles'][name]
    assert len(names)==62 and len(set(types))==9
    packet=dict(profiles=profiles,controller_order=names,types=dict(zip(names,types)),eta=pe['eta'],native_manifest=pe['native_manifest'])
    write(HERE/'BASE_PROTOCOL.json',packet)
    source_features={};source_r5={}
    for cohort in ('DEVELOPMENT','LOCKED'):
        with np.load(OLD/f'{cohort}_FEATURES.npz',allow_pickle=False) as z:
            for name,x,r in zip(z['names'],z['features'],z['r5']):source_features[str(name)]=x;source_r5[str(name)]=r
    np.savez_compressed(HERE/'CONTROLLER_FEATURES.npz',names=names,types=types,features=[source_features[n] for n in names],r5=[source_r5[n] for n in names])
    norm=read(LEARN/'normalization.json');n=norm['scenarios']['ring_exchange']
    dims={sc:len(v['h_mean']) for sc,v in norm['scenarios'].items()};cdim=len(norm['environment_keys'])+3
    gm,cm=Generator(),Critic()
    gp=serialization.from_bytes(merge_initialized(gm,dims,cdim),(LEARN/'generator/seed41/checkpoint.msgpack').read_bytes())
    cp=serialization.from_bytes(merge_initialized(cm,dims,cdim,critic=True),(LEARN/'critic/seed23/checkpoint.msgpack').read_bytes())
    reference=ReadOnlyConditioning();records=[];raw_h=[];raw_c=[];encoded=[]
    for index,s in enumerate(train+fresh):
        h,descriptor=reference.conditioning(dict(uid=s['state_uid'],physical=s['physical']))
        assert len(h)==100 and np.isfinite(h).all()
        c=np.zeros(cdim,np.float32);values=scalar_environment_fields(descriptor)
        for j,key in enumerate(norm['environment_keys']):c[j]=values.get(key,0.)
        c[-1]=1.;raw_h.append(h);raw_c.append(c)
        hh=(np.asarray(h,np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32)
        cc=(c-np.asarray(n['c_mean'],np.float32))/np.asarray(n['c_std'],np.float32)
        raw=np.asarray(gm.apply(gp,jnp.asarray(hh[None]),jnp.asarray(cc[None]),method=gm.ring))[0]
        proposal_seed=stable_int('generator-v1-proposals',41,'ring_exchange',s['state_uid'])
        rng=np.random.default_rng(proposal_seed)
        mean=np.asarray(eta_mean(jnp.asarray(raw)),float)
        samples=np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(4,0),jnp.asarray(rng.standard_normal((4,3)),jnp.float32)),float)
        etas=np.asarray([mean,*samples]);actions=[]
        for e in etas:actions.extend([action(e,e,'fixed'),action(e,pe['eta'][10],'progress')])
        actions.extend([action([0,0,0],[0,0,0],'fixed',True),action(pe['eta'][13],pe['eta'][10],'step500'),action(pe['eta'][13],pe['eta'][10],'progress')])
        vectors=[]
        for a in actions:
            onehot=[float(a['switch_mode']==m) for m in ('fixed','progress','step500')]
            vectors.append([*((np.asarray(a['eta_a'])-CENTER)/RADIUS),*((np.asarray(a['eta_b'])-CENTER)/RADIUS),*onehot,float(a['safety_baseline'])])
        encoded.append(vectors)
        logits=np.asarray(cm.apply(cp,jnp.asarray(np.repeat(hh[None],5,0)),jnp.asarray(np.repeat(cc[None],5,0)),jnp.asarray((etas-CENTER)/RADIUS),method=cm.ring))
        score=1/(1+np.exp(-logits));pick=int(np.argmax(score))
        records.append(dict(state_uid=s['state_uid'],state_index=index,split='training' if index<48 else 'new_evaluation',
            proposal_seed=proposal_seed,proposals=etas,actions=[dict(name=name,**a) for name,a in zip(ACTION_NAMES,actions)],
            original_critic_scores=score,original_pipeline_choice=pick*2,generator_only_choice=1))
    np.savez_compressed(HERE/'STATE_INPUTS.npz',h=np.asarray(raw_h,np.float32),c=np.asarray(raw_c,np.float32),
        action=np.asarray(encoded,np.float32),state_uids=[s['state_uid'] for s in train+fresh],split=['training']*48+['new_evaluation']*24)
    write(HERE/'CANDIDATES.json',dict(utc=utc(),action_names=ACTION_NAMES,states=records,proposals_controller_independent=True,
        generator_checkpoint_sha256=sha(LEARN/'generator/seed41/checkpoint.msgpack'),original_critic_checkpoint_sha256=sha(LEARN/'critic/seed23/checkpoint.msgpack')))
    seal('CANDIDATES.receipt.json',[HERE/name for name in ('CANDIDATES.json','STATE_INPUTS.npz','TRAINING_STATES.json','NEW_EVAL_STATES.json',
        'NEW_STATE_DEDUP.json','BASE_PROTOCOL.json','CONTROLLER_FEATURES.npz','condition_candidates_v2.py','frozen_networks.py')],
        before_any_new_evaluation_rollout=True,new_rollouts=0)
    print('Frozen 13 generator/anchor actions for 48 training and 24 new evaluation states',flush=True)


if __name__=='__main__':main()
