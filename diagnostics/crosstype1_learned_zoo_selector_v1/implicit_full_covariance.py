"""Three-round, 1024-sample CEM with the full fitted Gaussian covariance."""
import jax
import jax.numpy as jnp


def energy_candidates(net,parameters,obs,key,action_mean,action_scale):
    mean=jnp.zeros(8);covariance=jnp.eye(8);best=jnp.zeros(8);best_energy=jnp.inf
    for round_index in range(3):
        k=jax.random.fold_in(key,round_index)
        if round_index==0:
            raw=jax.random.uniform(k,(1024,8),minval=-1.,maxval=1.)
        else:
            factor=jnp.linalg.cholesky(covariance+1e-6*jnp.eye(8))
            raw=jnp.clip(mean+jax.random.normal(k,(1024,8))@factor.T,-1.,1.)
        normalized=(raw-action_mean)/action_scale
        energy=net.apply(parameters,jnp.broadcast_to(obs,(1024,obs.shape[-1])),normalized)
        index=jnp.argmin(energy);improve=energy[index]<best_energy
        best=jnp.where(improve,raw[index],best);best_energy=jnp.minimum(best_energy,energy[index])
        elite=raw[jnp.argsort(energy)[:103]];mean=elite.mean(0);centered=elite-mean
        covariance=centered.T@centered/len(elite)
    return (best-action_mean)/action_scale
