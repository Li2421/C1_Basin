"""GPU tests, including the required held-out energy-policy synthetic mode gate."""
from common import *
from zoo_models import network,initialize,PolicySession,infonce,energy_candidates
import jax
import jax.numpy as jnp
import optax
jax.config.update('jax_enable_x64',False)


def main():
    assert all(d.platform=='gpu' for d in jax.devices())
    key=jax.random.PRNGKey(2026101802)
    normal=dict(obs_mean=[0.]*92,obs_scale=[1.]*92,act_mean=[0.]*8,act_scale=[1.]*8,max_speed=.52)
    obs=np.random.default_rng(2026101802).normal(size=(4,23)).astype(np.float32)
    checks=[]
    for family in ('chunking','recurrent','cvae','implicit'):
        net=network(family);pa=initialize(net,family,key)
        session=PolicySession(family,pa,normal)
        a=session.action(obs,key,0)
        session.reset();b=session.action(obs,key,0)
        assert a.tobytes()==b.tobytes()
        assert a.shape==(4,2) and np.isfinite(a).all()
        assert np.linalg.norm(a,axis=-1).max()<=.52+1e-6
        if family=='chunking':
            saved=session.sequence.copy()
            # A caller may project/mutate its own output, never the cached chunk.
            b[:]=0
            assert session.sequence.tobytes()==saved.tobytes()
            for step in (1,2,3):
                next_action=session.action(obs*0,key,step)
                assert session.sequence_start==0
                assert next_action.tobytes()==saved[step].tobytes()
            session.action(obs*0,key,4)
            assert session.sequence_start==4
        if family=='recurrent':
            assert np.any(session.hidden!=0)
            session.action(obs,key,1)
            session.reset()
            assert np.all(session.hidden==0)
            assert session.action(obs,key,0).tobytes()==a.tobytes()
        checks.append(dict(family=family,key_reproducible=True,speed_bounded=True,passed=True))
    # Synthetic training is not controller training and uses no Ring outcomes.
    net=network('implicit');pa=initialize(net,'implicit',key);opt=optax.adam(3e-4);state=opt.init(pa)
    @jax.jit
    def block(p,st,steps):
        def update(carry,step):
            p,st=carry;k=jax.random.fold_in(key,step);km,kn,kl=jax.random.split(k,3)
            mode=jax.random.bernoulli(km,.5,(32,1))*2-1
            y=.6*mode+jax.random.normal(kn,(32,8))*.03
            def loss(pp):return infonce(net,pp,jnp.zeros((32,92)),y,kl,jnp.zeros(8),jnp.ones(8))
            value,grad=jax.value_and_grad(loss)(p);delta,st=opt.update(grad,st,p)
            return (optax.apply_updates(p,delta),st),value
        return jax.lax.scan(update,(p,st),steps)
    (pa,state),losses=block(pa,state,jnp.arange(2000))
    sample=jax.jit(lambda k:energy_candidates(net,pa,jnp.zeros(92),k,jnp.zeros(8),jnp.ones(8)))
    actions=np.stack([np.asarray(sample(jax.random.fold_in(key,3000+i))) for i in range(32)])
    negative=int(np.sum(np.linalg.norm(actions+.6,axis=-1)<.7))
    positive=int(np.sum(np.linalg.norm(actions-.6,axis=-1)<.7))
    mode_passed=min(negative,positive)>=3 and negative+positive>=26
    write(HERE/'IMPLICIT_SYNTHETIC_TEST.json',dict(passed=mode_passed,negative_mode=negative,positive_mode=positive,
          samples=32,training_steps=2000,criterion='>=3 samples in each mode and >=26/32 near either mode',
          training_seed=2026101802,synthetic_actions=actions,first_loss=float(losses[0]),last_loss=float(losses[-1]),
          no_ring_labels=True))
    # A failed implicit test affects the locked implicit type only.
    write(HERE/'ZOO_UNIT_TESTS.json',dict(utc=utc(),passed=all(c['passed'] for c in checks),
          checks=checks,implicit_mode_passed=mode_passed,physical_steps=0,x64=False))
    print('policy tests passed; implicit synthetic gate',mode_passed,flush=True)

if __name__=='__main__':main()
