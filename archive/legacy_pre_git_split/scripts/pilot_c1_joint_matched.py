"""Small matched pilots, gated by actual JAX derivative checks."""
import argparse,json,pickle,time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from audit_c1_joint_bptt import ROOT,OUT,setup,noise_for
from single_integrator.c1.joint_frozen_rollout import rollout

DEST=OUT/'pilot'

def prepare():
    DEST.mkdir(exist_ok=True);rng=np.random.default_rng(20261020);x=rng.uniform(.55,1.05,4);y=rng.uniform(-.025,.025,(4,2))
    initial=np.stack([np.c_[-x,y[:,0]],np.c_[x,y[:,1]]],axis=1)
    p=dict(initials=initial.tolist(),ids=list(range(200000,200004)),seed=0,updates=8,batch_size=1,lr=1e-5,
        architecture=[256,256,256],loss='same-state deviation + selected risk (fixed common dual=1); P or frozen full. No risk rescaling.',
        optimization='Adam; same-batch backtracking up to8 halvings; identical cyclic order and noise for both arms.',
        training_horizon='0..5s frozen prefix, trainable5..25s, score5..15s with progress endpoint25s; success absorbing, deadlock continues.',
        evaluation_ids=list(range(100000,100016)),evaluation_seeds=[20260916,20261001],
        scope='Eight-step engineering pilot; insufficient for final training efficacy claims.')
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:path.write_text(json.dumps(p,indent=2)+'\n')

def train(name):
    gate=json.loads((OUT/'gradient_gate.json').read_text());assert gate['pass_for_small_pilot']
    p=json.loads((DEST/'protocol.json').read_text());params,field,plant,cbf=setup();optimizer=optax.adam(p['lr']);state=optimizer.init(params)
    initials=jnp.asarray(p['initials']);noises=jnp.stack([noise_for(rid) for rid in p['ids']]);idx=0 if name=='P' else 3
    fn=jax.jit(lambda phi,x,n:rollout(phi,field,x,n,plant,cbf)[:2])
    def objective(phi,x,n):
        terms,deviation=fn(phi,x,n);return terms[idx]+deviation,(terms,deviation)
    vg=jax.value_and_grad(objective,has_aux=True);logs=[];started=time.time()
    for step in range(p['updates']):
        i=step%len(initials);(loss,(terms,dev)),grad=vg(params,initials[i],noises[i]);norm=float(optax.global_norm(grad))
        if not np.isfinite(norm):raise FloatingPointError('Nonfinite pilot gradient')
        update,newstate=optimizer.update(grad,state,params);accepted=False;post=float(loss)
        for backtrack in range(9):
            proposal=optax.apply_updates(params,jax.tree_util.tree_map(lambda x:x*2.**(-backtrack),update))
            try:
                value,_=objective(proposal,initials[i],noises[i]);post=float(value)
                if np.isfinite(post) and post<=float(loss)+1e-10:accepted=True;params=proposal;state=newstate;break
            except Exception:pass
        row=dict(step=step,rid=p['ids'][i],pre=float(loss),post=post,gradient_norm=norm,terms=np.asarray(terms).tolist(),deviation=float(dev),accepted=accepted,backtracks=backtrack)
        logs.append(row);(DEST/f'{name}_history.json').write_text(json.dumps(logs,indent=2)+'\n');print(name,row,flush=True)
    (DEST/f'{name}_params.pkl').write_bytes(pickle.dumps(jax.tree_util.tree_map(np.asarray,params)))
    final=[np.asarray(fn(params,initials[i],noises[i])[0]).tolist() for i in range(4)]
    (DEST/f'{name}_summary.json').write_text(json.dumps(dict(final_training_terms=final,elapsed_seconds=time.time()-started),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--arm',choices=['P','full']);a=parser.parse_args()
    if a.prepare:prepare()
    else:train(a.arm)
