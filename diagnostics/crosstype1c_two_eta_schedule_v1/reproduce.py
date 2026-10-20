"""Mandatory cross-type1b execution and generator prefix gates."""
from common import *
OLD=D/'crosstype1b_generator_critic_pipeline_v1'
def execution_design():
    prereg();base=read(OLD/'BASE_PROTOCOL.json');cases=read(OLD/'new_evaluation_CASES.json')
    desired={'Flow':2,'gaussian':2,'deterministic':2,'mixture':2,'diffusion':2,'chunking':2,'recurrent':2,'cvae':1,'ring_ippo':1}
    rows=[]
    for kind,count in desired.items():
        names=[n for n in base['controller_order'] if base['types'][n]==kind]
        if kind=='Flow':names=['base_u_v10_local_macflow',next(n for n in names if n!='base_u_v10_local_macflow')]
        for i,name in enumerate(names[:count]):
            candidate=0 if i%2 else 1
            c=next(c for c in cases if c['base_controller_name']==name and c['seed']==0 and c['candidate_index']==candidate)
            ref=OLD/'new_evaluation/raw'/f"{c['rollout_uid']}.json";r=read(ref)
            assert sha(r['trace_path'])==r['trace_sha256']
            rows.append(dict(kind=kind,case=c,reference=str(ref),reference_sha256=sha(ref),protocol=str(OLD/'new_evaluation_protocol.json')))
    assert len(rows)==16
    write(HERE/'REPRODUCTION_CASES.json',rows)
    path=HERE/'reproduction_PLANNED_ROLLOUTS.json'
    write(path,dict(requests=[{k:r['case'][k] for k in ('state_uid','eta_uid','controller_uid','seed_key')} for r in rows]))
    write(HERE/'reproduction_CACHE_PREFLIGHT.json',planner.preflight(path))
    seal('REPRODUCTION_DESIGN.receipt.json',[HERE/'REPRODUCTION_CASES.json',path,HERE/'reproduction_CACHE_PREFLIGHT.json'],mandatory_replay_not_new_labels=True,selection_outcome_blind=True)
def execution():
    from generated_execution import execute
    rows=read(HERE/'REPRODUCTION_CASES.json');checks=[]
    for i,r in enumerate(rows):
        assert sha(r['reference'])==r['reference_sha256']
        ref=read(r['reference']);p=read(r['protocol']);out=execute(r['case'],p,{},'reproduction')
        write(HERE/f'reproduction/raw/{i:02}.json',out)
        with np.load(ref['trace_path'],allow_pickle=False) as a,np.load(out['trace_path'],allow_pickle=False) as b:
            x,y=a['positions'],b['positions'];positions=x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
        first=np.asarray(ref['first_action'],np.float64).tobytes()==np.asarray(out['first_action'],np.float64).tobytes()
        c=dict(type=r['kind'],controller=r['case']['base_controller_name'],candidate=r['case']['candidate_index'],first_action_bytes=first,positions_bytes=positions,termination=ref['outcome']==out['outcome'],steps=ref['episode_length']==out['episode_length'],x64=out['jax_enable_x64'])
        c['passed']=all(c[k] for k in ('first_action_bytes','positions_bytes','termination','steps')) and not c['x64'];checks.append(c);print(json.dumps(c),flush=True)
    report=dict(utc=utc(),passed=all(c['passed'] for c in checks),checks=checks,count=16,types=9,new_task_labels=0)
    write(HERE/'EXECUTION_REPRODUCTION.json',report);seal('EXECUTION_REPRODUCTION.receipt.json',[HERE/'EXECUTION_REPRODUCTION.json',HERE/'reproduce.py',HERE/'BOOTSTRAP_SOURCES.json'])
    assert report['passed'],'Mandatory reproduction failed: stop the entire experiment'
def generator():
    from legacy_conditioning import ReadOnlyConditioning
    from frozen_networks import Generator,merge_initialized,eta_mean,eta_from_noise,stable_int,scalar_environment_fields
    from flax import serialization
    import jax,jax.numpy as jnp
    jax.config.update('jax_enable_x64',False)
    norm=read(D/'orthoflow3_generator_critic_v1/normalization.json');n=norm['scenarios']['ring_exchange'];dims={k:len(v['h_mean']) for k,v in norm['scenarios'].items()};cdim=len(norm['environment_keys'])+3
    gm=Generator();weight=D/'orthoflow3_generator_critic_v1/generator/seed41/checkpoint.msgpack';parameters=serialization.from_bytes(merge_initialized(gm,dims,cdim),weight.read_bytes())
    reference=ReadOnlyConditioning();states=read(OLD/'TRAINING_STATES.json')+read(OLD/'NEW_EVAL_STATES.json');expected=read(OLD/'CANDIDATES.json')['states'];checks=[]
    with np.load(OLD/'STATE_INPUTS.npz',allow_pickle=False) as z:old_h=z['h'];old_c=z['c']
    for i,(s,e) in enumerate(zip(states,expected)):
        h,descriptor=reference.conditioning(dict(uid=s['state_uid'],physical=s['physical']));c=np.zeros(cdim,np.float32);values=scalar_environment_fields(descriptor)
        for j,k in enumerate(norm['environment_keys']):c[j]=values.get(k,0.)
        c[-1]=1.;hh=(np.asarray(h,np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32);cc=(c-np.asarray(n['c_mean'],np.float32))/np.asarray(n['c_std'],np.float32)
        raw=np.asarray(gm.apply(parameters,jnp.asarray(hh[None]),jnp.asarray(cc[None]),method=gm.ring))[0]
        rng=np.random.default_rng(stable_int('generator-v1-proposals',41,'ring_exchange',s['state_uid']));noise=rng.standard_normal((7,3)).astype(np.float32)
        samples=np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(7,0),jnp.asarray(noise)),float);etas=np.asarray([np.asarray(eta_mean(jnp.asarray(raw)),float),*samples])
        prior=np.asarray(e['proposals'],np.float64)
        checks.append(dict(state_uid=s['state_uid'],proposals_bitwise=etas[:5].tobytes()==prior.tobytes(),h_bitwise=np.asarray(h,np.float32).tobytes()==old_h[i].tobytes(),c_bitwise=c.tobytes()==old_c[i].tobytes(),max_abs_error=float(np.max(abs(etas[:5]-prior)))))
    result=dict(utc=utc(),passed=len(checks)==72 and all(c['proposals_bitwise'] and c['h_bitwise'] and c['c_bitwise'] for c in checks),states=checks,generator_checkpoint_sha256=sha(weight),x64=False,backend=jax.default_backend(),new_rollouts=0)
    write(HERE/'GENERATOR_REPRODUCTION.json',result);seal('GENERATOR_REPRODUCTION.receipt.json',[HERE/'GENERATOR_REPRODUCTION.json',HERE/'reproduce.py',HERE/'frozen_networks.py',HERE/'legacy_conditioning.py'])
    print('All72 generator prefix and conditioning bitwise gate:',result['passed'],flush=True)
    assert result['passed'],'Mandatory generator reproduction failed: stop the entire experiment'
if __name__=='__main__':globals()[sys.argv[1]]()
