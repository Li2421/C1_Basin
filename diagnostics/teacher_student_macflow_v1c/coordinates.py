"""JAX row-wise radial/tangential transform matching ring_exchange.local_frame.frames."""
from core import *
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',False)
def to_local_jax(vectors,positions):
    radial=positions/jnp.maximum(jnp.linalg.norm(positions,axis=-1,keepdims=True),1e-12)
    tangent=jnp.stack((-radial[...,1],radial[...,0]),axis=-1)
    return jnp.stack((jnp.sum(vectors*radial,axis=-1),jnp.sum(vectors*tangent,axis=-1)),axis=-1)
