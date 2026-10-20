"""Matched no-state/no-controller branch for the shrinkage critic."""
from critic_model import *
from critic_model import network as base_network,initialize as base_initialize

MODES=('action_only',)
def network(mode):
    return Critic(use_state=False,use_controller=False) if mode=='action_only' else base_network(mode)

def initialize(mode,seed,initialization,transform,original=None):
    if mode!='action_only':return base_initialize(mode,seed,initialization,transform,original)
    if initialization=='scratch':
        return network(mode).init(jax.random.PRNGKey(seed),jnp.zeros((1,100)),jnp.zeros((1,33)),jnp.zeros((1,24)),jnp.zeros((1,10)))
    p=unfreeze(base_initialize('no_state',seed,initialization,transform,original))
    del p['params']['controller_encoder']
    p['params']['shared1']['kernel']=p['params']['shared1']['kernel'][:32]
    return p
