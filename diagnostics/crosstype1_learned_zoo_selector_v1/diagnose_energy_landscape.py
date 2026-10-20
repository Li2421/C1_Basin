"""Inference-only diagnosis of the preserved synthetic checkpoint and CEM."""
from common import *
from zoo_models import network,initialize
from implicit_full_covariance import energy_candidates
from flax import serialization
import jax
import jax.numpy as jnp


class AnalyticModes:
    def apply(self,parameters,obs,actions):
        return jnp.minimum(jnp.sum((actions-.6)**2,axis=-1),jnp.sum((actions+.6)**2,axis=-1))


def main():
    jax.config.update('jax_enable_x64',False)
    assert all(d.platform=='gpu' for d in jax.devices())
    receipt=read(HERE/'IMPLICIT_FULLCOV_SYNTHETIC_TEST.json')
    assert sha(receipt['checkpoint'])==receipt['checkpoint_sha256']
    key=jax.random.PRNGKey(2026101802);net=network('implicit')
    params=serialization.from_bytes(initialize(net,'implicit',key),Path(receipt['checkpoint']).read_bytes())
    grid=jnp.linspace(-1,1,101)
    energies=np.asarray(net.apply(params,jnp.zeros((101,92)),jnp.broadcast_to(grid[:,None],(101,8))))
    rows={}
    for name,model in [('trained_energy',net),('analytic_two_modes',AnalyticModes())]:
        sample=jax.jit(lambda k:energy_candidates(model,params,jnp.zeros(92),k,jnp.zeros(8),jnp.ones(8)))
        actions=np.stack([np.asarray(sample(jax.random.fold_in(key,3000+i))) for i in range(32)])
        neg=int(np.sum(np.linalg.norm(actions+.6,axis=1)<.7));pos=int(np.sum(np.linalg.norm(actions-.6,axis=1)<.7))
        rows[name]=dict(negative_mode=neg,positive_mode=pos,passed=min(neg,pos)>=3 and neg+pos>=26,actions=actions)
    write(HERE/'IMPLICIT_LANDSCAPE_DIAGNOSIS.json',dict(utc=utc(),no_training=True,no_ring_labels=True,
        checkpoint_sha256=receipt['checkpoint_sha256'],diagonal_coordinate=np.asarray(grid),diagonal_energy=energies,
        minimum_coordinate=float(grid[np.argmin(energies)]),mode_energy_at_negative_point=float(energies[20]),
        mode_energy_at_positive_point=float(energies[80]),samplers=rows))
    print({k:{a:b for a,b in v.items() if a!='actions'} for k,v in rows.items()},flush=True)


if __name__=='__main__':main()
