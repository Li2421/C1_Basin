"""Recompute original frozen Ring validation proposals and critic scores."""
from common import *
from frozen_networks import Generator,Critic,merge_initialized,eta_mean,eta_from_noise,dist_params,stable_int,CENTER,RADIUS,scalar_environment_fields
from flax import serialization
import jax,jax.numpy as jnp
import pyarrow.parquet as pq
jax.config.update('jax_enable_x64',False)

LEARN=D/'orthoflow3_generator_critic_v1'


def main():
    prereg();assert not (HERE/'FROZEN_COMPONENT_REPRODUCTION.json').exists()
    norm=read(LEARN/'normalization.json');n=norm['scenarios']['ring_exchange']
    dims={sc:len(v['h_mean']) for sc,v in norm['scenarios'].items()};cdim=len(norm['environment_keys'])+3
    gm,cm=Generator(),Critic()
    gp_path=LEARN/'generator/seed41/checkpoint.msgpack';cp_path=LEARN/'critic/seed23/checkpoint.msgpack'
    assert sha(gp_path)=='a7bb0476cf25b7e0046a5d977d73f91aa478191c3f59ca73a5c487d4bf5d9965'
    assert sha(cp_path)=='f0a2dd358a477590cf434c7803d6ddcc307509157a9698c7dbfaafcf302e02ed'
    gp=serialization.from_bytes(merge_initialized(gm,dims,cdim),gp_path.read_bytes())
    cp=serialization.from_bytes(merge_initialized(cm,dims,cdim,critic=True),cp_path.read_bytes())
    data_path=ROOT/'datasets/orthoflow3_basin_dataset_v1/states.parquet'
    rows=pq.read_table(data_path,filters=[('scenario','=','ring_exchange')]).to_pylist()
    by={r['state_uid']:r for r in rows}
    proposals=read(LEARN/'generator/frozen_validation_proposals.json')
    scores={r['state_uid']:r for r in read(LEARN/'critic/frozen_validation_selection.json')['states'] if r['scenario']=='ring_exchange'}
    checks=[]
    for reference in proposals['states']:
        if reference['scenario']!='ring_exchange':continue
        r=by[reference['state_uid']]
        conditioning=json.loads(r['conditioning']) if isinstance(r['conditioning'],str) else r['conditioning']
        descriptor=json.loads(r['environment_descriptor']) if isinstance(r['environment_descriptor'],str) else r['environment_descriptor']
        h=(np.asarray(conditioning['flat'],np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32)
        values=scalar_environment_fields(descriptor);c=np.zeros(cdim,np.float32)
        for j,key in enumerate(norm['environment_keys']):c[j]=values.get(key,0.)
        c[-1]=1.;c=(c-np.asarray(n['c_mean'],np.float32))/np.asarray(n['c_std'],np.float32)
        raw=np.asarray(gm.apply(gp,jnp.asarray(h[None]),jnp.asarray(c[None]),method=gm.ring))[0]
        rng=np.random.default_rng(stable_int('generator-v1-proposals',41,'ring_exchange',r['state_uid']))
        mean=np.asarray(eta_mean(jnp.asarray(raw)))
        samples=np.asarray(eta_from_noise(jnp.asarray(raw)[None].repeat(4,0),jnp.asarray(rng.standard_normal((4,3)),jnp.float32)))
        eta=np.asarray([mean.astype(float),*samples.astype(float)])
        logits=np.asarray(cm.apply(cp,jnp.asarray(np.repeat(h[None],5,0)),jnp.asarray(np.repeat(c[None],5,0)),jnp.asarray((eta-CENTER)/RADIUS),method=cm.ring))
        prediction=1/(1+np.exp(-logits));original=np.asarray(scores[r['state_uid']]['critic_scores'])
        expected=np.asarray([reference['generator_mean'],*reference['samples']])
        proposal_error=float(np.max(np.abs(eta-expected)));score_error=float(np.max(np.abs(prediction-original)))
        checks.append(dict(state_uid=r['state_uid'],proposal_bitwise=eta.tobytes()==expected.tobytes(),
            proposal_max_abs_error=proposal_error,critic_bitwise=prediction.astype(float).tobytes()==original.tobytes(),
            critic_max_abs_error=score_error,critic_choice_matches=int(np.argmax(prediction))==scores[r['state_uid']]['critic_index'],
            generator_passed=proposal_error<=1e-6,critic_exactly_passed=score_error==0.))
    assert len(checks)==16
    record=dict(utc=utc(),states=checks,generator_passed=all(c['generator_passed'] for c in checks),
        critic_bitwise_passed=all(c['critic_exactly_passed'] and c['critic_choice_matches'] for c in checks),
        x64=False,backend=jax.default_backend(),new_rollouts=0,source_inputs={str(f):sha(f) for f in
            (gp_path,cp_path,LEARN/'normalization.json',data_path,LEARN/'generator/frozen_validation_proposals.json',LEARN/'critic/frozen_validation_selection.json')})
    record['passed']=record['generator_passed'] and record['critic_bitwise_passed']
    write(HERE/'FROZEN_COMPONENT_REPRODUCTION.json',record)
    seal('FROZEN_COMPONENT_REPRODUCTION.receipt.json',[HERE/'FROZEN_COMPONENT_REPRODUCTION.json',HERE/'reproduce_frozen.py',HERE/'frozen_networks.py'],passed=record['passed'])
    print(json.dumps(dict(passed=record['passed'],generator_max_error=max(c['proposal_max_abs_error'] for c in checks),critic_max_error=max(c['critic_max_abs_error'] for c in checks))),flush=True)


if __name__=='__main__':main()
