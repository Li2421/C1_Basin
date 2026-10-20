"""Same synthetic training/criterion, paired diagonal-vs-full-covariance inference."""
from common import *
from zoo_models import network,initialize,infonce,energy_candidates as diagonal
from implicit_full_covariance import energy_candidates as full
from flax import serialization
import jax
import jax.numpy as jnp
import optax

def main():
    jax.config.update('jax_enable_x64',False);assert all(d.platform=='gpu' for d in jax.devices())
    key=jax.random.PRNGKey(2026101802);net=network('implicit');pa=initialize(net,'implicit',key)
    optimizer=optax.adam(3e-4);state=optimizer.init(pa)
    @jax.jit
    def fit(p,st):
        def update(carry,step):
            p,st=carry;k=jax.random.fold_in(key,step);km,kn,kl=jax.random.split(k,3)
            mode=jax.random.bernoulli(km,.5,(32,1))*2-1;y=.6*mode+jax.random.normal(kn,(32,8))*.03
            def loss(pp):return infonce(net,pp,jnp.zeros((32,92)),y,kl,jnp.zeros(8),jnp.ones(8))
            value,grad=jax.value_and_grad(loss)(p);change,st=optimizer.update(grad,st,p)
            return (optax.apply_updates(p,change),st),value
        return jax.lax.scan(update,(p,st),jnp.arange(2000))
    (pa,state),losses=fit(pa,state)
    folder=HERE/'implicit_synthetic_diagnosis';folder.mkdir(exist_ok=True)
    (folder/'model.msgpack').write_bytes(serialization.to_bytes(pa))
    outputs={}
    for name,sampler in [('diagonal',diagonal),('full_covariance',full)]:
        sample=jax.jit(lambda k:sampler(net,pa,jnp.zeros(92),k,jnp.zeros(8),jnp.ones(8)))
        actions=np.stack([np.asarray(sample(jax.random.fold_in(key,3000+i))) for i in range(32)])
        negative=int(np.sum(np.linalg.norm(actions+.6,axis=-1)<.7));positive=int(np.sum(np.linalg.norm(actions-.6,axis=-1)<.7))
        outputs[name]=dict(passed=min(negative,positive)>=3 and negative+positive>=26,
            negative_mode=negative,positive_mode=positive,actions=actions)
    anchors=jnp.array([[-.6]*8,[0.]*8,[.6]*8]);energies=net.apply(pa,jnp.zeros((3,92)),anchors)
    write(HERE/'IMPLICIT_FULLCOV_SYNTHETIC_TEST.json',dict(utc=utc(),passed=outputs['full_covariance']['passed'],
          candidates=outputs,anchor_energies=np.asarray(energies),training_steps=2000,training_seed=2026101802,
          first_loss=float(losses[0]),last_loss=float(losses[-1]),criterion_unchanged='>=3 samples each mode; >=26/32 within Euclidean .7',
          no_ring_labels=True,checkpoint=str(folder/'model.msgpack'),checkpoint_sha256=sha(folder/'model.msgpack')))
    print({name:{k:v for k,v in result.items() if k!='actions'} for name,result in outputs.items()},flush=True)

if __name__=='__main__':main()
