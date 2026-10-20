"""Restore frozen C5 model at original14/24 dimensions and its full415 menu."""
from common import *
from legacy_critic_model import Critic,transform_arrays
from reference_candidate_menu import menu,choose,METHODS as C5_METHODS
from action_encoding import encode
from flax import serialization
import jax,jax.numpy as jnp

def predict(model,data):
    if model.get('kind')=='probability_shrinkage':
        w=model['full_weight'];return w*predict(model['full'],data)+(1-w)*predict(model['no_controller'],data)
    for kind in ('checkpoint','transform','fit_receipt'):assert sha(model[kind+'_path'])==model[kind+'_sha256']
    t=read(model['transform_path'])
    for k in ('h_mean','h_scale','c_mean','c_scale'):t[k]=np.asarray(t[k],np.float32)
    for k in ('feature_fill','feature_mean','feature_scale'):t[k]=np.asarray(t[k],np.float64)
    net=Critic(use_state=True,use_controller=model['mode']!='no_controller');template=jax.tree_util.tree_map(lambda *x:jnp.stack(x),*[net.init(jax.random.PRNGKey(s),jnp.zeros((1,100)),jnp.zeros((1,33)),jnp.zeros((1,24)),jnp.zeros((1,14))) for s in (17,23,41)]);parameters=serialization.from_bytes(template,Path(model['checkpoint_path']).read_bytes());arrays=transform_arrays(data,t);nc,ns,na=arrays['action'].shape[:3]
    h=jnp.broadcast_to(jnp.asarray(arrays['h'])[None,:,None,:],(nc,ns,na,100)).reshape(-1,100);c=jnp.broadcast_to(jnp.asarray(arrays['c'])[None,:,None,:],(nc,ns,na,33)).reshape(-1,33);C=jnp.broadcast_to(jnp.asarray(arrays['controller'])[:,None,None,:],(nc,ns,na,24)).reshape(-1,24);a=jnp.asarray(arrays['action']).reshape(-1,14)
    fn=jax.jit(jax.vmap(lambda p:jax.nn.sigmoid(net.apply(p,h,c,C,a)).reshape(nc,ns,na)))
    return np.asarray(fn(parameters))

def selected(held,data,proposals,state_uids):
    from reference_schedule_generator import predict as generate
    frozen=read(D/'crosstype5_repair_gate_late_eta_v1/FROZEN_PIPELINE.json')['outer'][held];assert held not in frozen['full']['source_types']
    pairs=generate(frozen['generator'],data,state_uids);nc=len(data['names']);ns=len(state_uids);menus=[[menu(proposals[si],pairs[ci,si]) for si in range(ns)] for ci in range(nc)];vectors=np.zeros((nc,ns,415,14),np.float32)
    for ci in range(nc):
        for si in range(ns):vectors[ci,si,:len(menus[ci][si])]=[encode(row['action']) for row in menus[ci][si]]
    probability=predict(frozen['full'],dict(data,action=vectors));g=frozen['gates'];t=read(frozen['full']['full']['transform_path']);filled=np.where(np.isfinite(data['features']),data['features'],t['feature_fill']);out=[]
    for ci in range(nc):
        rr=[]
        for si in range(ns):
            rows=menus[ci][si];chosen,details=choose(rows,probability[:,ci,si,:len(rows)],g['critic']['k'],g['critic']['delta'],g['probe']['tau'],g['probe']['rho'],filled[ci]);a=rows[chosen[C5_METHODS[0]]]['action'];rr.append(dict(a,c6_schedule=False))
        out.append(rr)
    return out
