from common import *
OLD=D/'crosstype1c_two_eta_schedule_v1'
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
        checks.append(dict(state_uid=s['state_uid'],proposals_bitwise=etas.tobytes()==prior.tobytes(),h_bitwise=np.asarray(h,np.float32).tobytes()==old_h[i].tobytes(),c_bitwise=c.tobytes()==old_c[i].tobytes(),max_abs_error=float(np.max(abs(etas-prior)))))
    result=dict(utc=utc(),passed=len(checks)==72 and all(c['proposals_bitwise'] and c['h_bitwise'] and c['c_bitwise'] for c in checks),states=checks,generator_checkpoint_sha256=sha(weight),x64=False,backend=jax.default_backend(),new_rollouts=0)
    write(HERE/'GENERATOR_REPRODUCTION.json',result);seal('GENERATOR_REPRODUCTION.receipt.json',[HERE/'GENERATOR_REPRODUCTION.json',HERE/'generator_gate.py',HERE/'frozen_networks.py',HERE/'legacy_conditioning.py'])
    print('All72 eight-proposal and conditioning bitwise gate:',result['passed'],flush=True)
    assert result['passed'],'Mandatory generator reproduction failed: stop the entire experiment'

if __name__=='__main__':generator()
