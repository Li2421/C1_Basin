"""Diagnose unbounded ReLU extrapolation without changing any Ring runtime."""
from common import *
from zoo_models import infonce
from energy_activation_candidates import SmoothEnergy
from implicit_full_covariance import energy_candidates
from flax import serialization
import jax
import jax.numpy as jnp
import optax


def main():
    jax.config.update('jax_enable_x64',False);assert all(d.platform=='gpu' for d in jax.devices())
    assert (HERE/'ENERGY_ACTIVATION_DESIGN.receipt.json').exists()
    key=jax.random.PRNGKey(2026101802)
    outputs={}
    for activation in ('tanh','silu','softplus'):
        net=SmoothEnergy(activation);pa=net.init(key,jnp.zeros((1,92)),jnp.zeros((1,8)))
        optimizer=optax.adam(3e-4);state=optimizer.init(pa)
        @jax.jit
        def fit(p,st):
            def update(carry,step):
                p,st=carry;km,kn,kl=jax.random.split(jax.random.fold_in(key,step),3)
                mode=jax.random.bernoulli(km,.5,(32,1))*2-1
                y=.6*mode+jax.random.normal(kn,(32,8))*.03
                def loss(pp):return infonce(net,pp,jnp.zeros((32,92)),y,kl,jnp.zeros(8),jnp.ones(8))
                value,grad=jax.value_and_grad(loss)(p);change,st=optimizer.update(grad,st,p)
                return (optax.apply_updates(p,change),st),value
            return jax.lax.scan(update,(p,st),jnp.arange(2000))
        (pa,state),losses=fit(pa,state)
        folder=HERE/'implicit_synthetic_diagnosis'/activation;folder.mkdir(exist_ok=True)
        (folder/'model.msgpack').write_bytes(serialization.to_bytes(pa))
        sample=jax.jit(lambda k:energy_candidates(net,pa,jnp.zeros(92),k,jnp.zeros(8),jnp.ones(8)))
        actions=np.stack([np.asarray(sample(jax.random.fold_in(key,3000+i))) for i in range(32)])
        neg=int(np.sum(np.linalg.norm(actions+.6,axis=1)<.7));pos=int(np.sum(np.linalg.norm(actions-.6,axis=1)<.7))
        grid=jnp.linspace(-1,1,101)
        energies=net.apply(pa,jnp.zeros((101,92)),jnp.broadcast_to(grid[:,None],(101,8)))
        outputs[activation]=dict(passed=min(neg,pos)>=3 and neg+pos>=26,negative_mode=neg,positive_mode=pos,
            actions=actions,first_loss=float(losses[0]),last_loss=float(losses[-1]),diagonal_coordinate=np.asarray(grid),
            diagonal_energy=np.asarray(energies),checkpoint=str(folder/'model.msgpack'),checkpoint_sha256=sha(folder/'model.msgpack'))
        write(HERE/'IMPLICIT_ACTIVATION_TEST.partial.json',outputs)
        print(activation,neg,pos,outputs[activation]['passed'],flush=True)
        jax.clear_caches()
    write(HERE/'IMPLICIT_ACTIVATION_TEST.json',dict(utc=utc(),candidates=outputs,no_ring_labels=True,
        criterion_unchanged=True,training_seed=2026101802,training_steps=2000,negatives=256,cem_rounds=3,cem_samples=1024))


if __name__=='__main__':main()
