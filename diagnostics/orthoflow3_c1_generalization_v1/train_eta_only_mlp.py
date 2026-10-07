"""Strong continuous eta-only MLP control under the identical frozen splits."""

import argparse
import json
from collections import defaultdict

import flax.linen as nn
import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from offline_baselines import OUT
from train_interaction import SCENARIOS, prepare


class EtaOnly(nn.Module):
    @nn.compact
    def __call__(self, eta):
        x=nn.silu(nn.Dense(64)(eta))
        x=nn.silu(nn.Dense(64)(x))
        return nn.Dense(1)(x)[...,0]


def evaluate(model,params,pool):
    logits=np.asarray(model.apply(params,jnp.asarray(pool['eta'])))
    p=np.clip(1/(1+np.exp(-logits)),1e-5,1-1e-5)
    y=pool['y']
    nll=float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))
    brier=float(np.mean((p-y)**2))
    by=defaultdict(list)
    for i,s in enumerate(pool['state_uids']):by[s].append(i)
    selected=sum(float(y[ix[int(np.argmax(p[ix]))]]) for ix in by.values())
    oracle=sum(float(np.max(y[ix])) for ix in by.values())
    return {'pairs':len(y),'states':len(by),'nll':nll,'brier':brier,
            'selected_b15_states':int(selected),'oracle_b15_states':int(oracle)}


def run(seed,sc,data):
    model=EtaOnly();params=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,3)))
    opt=optax.adamw(1e-3,weight_decay=1e-4);state=opt.init(params)
    rng=np.random.default_rng(seed+445551)
    pool=data[sc]['parts']

    @jax.jit
    def step(params,state,eta,y):
        def loss(pp):return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(pp,eta),y))
        value,grad=jax.value_and_grad(loss)(params)
        update,state=opt.update(grad,state,params)
        return optax.apply_updates(params,update),state,value

    best=(float('inf'),None,0);stale=0;history=[]
    for it in range(1,1801):
        ix=rng.integers(0,len(pool['train']['y']),size=256)
        params,state,loss=step(params,state,jnp.asarray(pool['train']['eta'][ix]),jnp.asarray(pool['train']['y'][ix]))
        if it%100:continue
        nll=evaluate(model,params,pool['dev'])['nll']
        history.append({'step':it,'train_loss':float(loss),'dev_nll':nll})
        if nll<best[0]-1e-5:best=(nll,serialization.to_bytes(params),it);stale=0
        else:stale+=1
        if stale>=6 and it>=600:break
    params=serialization.from_bytes(params,best[1])
    folder=OUT/'eta_only_mlp'/f'seed{seed}'/sc;folder.mkdir(parents=True,exist_ok=True)
    (folder/'checkpoint.msgpack').write_bytes(best[1])
    (folder/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    summary={'seed':seed,'scenario':sc,'best_step':best[2],'dev_nll':best[0],
             'metrics':{p:evaluate(model,params,pool[p]) for p in ('dev','spatial_dev','test')},
             'checkpoint':str(folder/'checkpoint.msgpack'),'new_rollout':0}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=(17,23,41));arg=p.parse_args()
    data=prepare();results=[run(arg.seed,sc,data) for sc in SCENARIOS]
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
