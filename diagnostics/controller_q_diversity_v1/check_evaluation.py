"""End-to-end synthetic falsification with known controller/state reversals."""
import json
import sqlite3
import numpy as np
from .experiment import HERE,write,sha
from .train import KINDS
from . import evaluate as ev


def run():
    root=HERE/'checks/evaluation_synthetic';(root/'data/final').mkdir(parents=True,exist_ok=True)
    (root/'data/source128').mkdir(parents=True,exist_ok=True)
    n=8;meta=[];success=[];unknown=[]
    for scene in ('ring_exchange','toy_give_way'):
        for s in range(n):
            for ci in range(2):
                meta.append(dict(scene=scene,state_uid=f'{scene}_synthetic_{s}',controller=f'analytic{6+ci}',split='final'))
                y=np.zeros((16,16),np.int8);u=np.zeros_like(y)
                preferred=(s>=4)^bool(ci);y[int(preferred)]=1
                if s==0:u[15]=1
                success.append(y);unknown.append(u)
    ss=np.asarray(success);uu=np.asarray(unknown);rows=len(ss)
    data={k:np.asarray([r[k] for r in meta]) for k in meta[0]}
    data.update(seed_success=ss,seed_unknown=uu,seed_collision=np.zeros_like(ss),
        success=ss.sum(-1),unknown=uu.sum(-1),failure=16-ss.sum(-1)-uu.sum(-1))
    np.savez_compressed(root/'data/final/dataset.npz',**data)
    source=dict(split=np.array(['train']),scene=np.array(['toy_give_way']),controller=np.array(['analytic0']),success=np.array([[1]+[0]*15]),
        failure=np.array([[0]+[1]*15]),unknown=np.zeros((1,16),int))
    np.savez_compressed(root/'data/source128/dataset.npz',**source)
    pred={k:data[k] for k in ('scene','state_uid','controller')};perfect=ss.mean(-1).astype(float)
    swap=np.arange(rows).reshape(-1,2)[:,::-1].ravel()
    for kind in KINDS:
        q=perfect.copy()
        if kind=='eta_only':q[:]=0;q[:,0]=1
        if kind=='state_only':q=q[np.arange(rows)//2*2]
        pred[kind]=np.repeat(q[None],3,axis=0)
        pred[kind+'_controller_mismatch']=pred[kind][:,swap]
    perms=[]
    for j in range(100):
        index=np.arange(rows).reshape(2,n,2)
        perms.append(np.roll(index,1+j%(n-1),axis=1).ravel())
    pred['state_permutations']=np.asarray(perms)
    np.savez_compressed(root/'confirmation_predictions.npz',**pred)
    write(root/'models_frozen.json',dict(primary='state_context',train_limit=128))
    write(root/'confirmation_predictions_frozen.json',dict(predictions_sha256=sha(root/'confirmation_predictions.npz')))
    dbpath=root/'safety.sqlite';con=sqlite3.connect(dbpath)
    con.execute('CREATE TABLE IF NOT EXISTS rollout(rollout_uid TEXT PRIMARY KEY,success INT,numerical_failure INT,collision INT)')
    cases=[]
    for i,r in enumerate(meta):
        for seed in range(16):
            key=f'synthetic_safety_{i}_{seed}'
            con.execute('INSERT OR REPLACE INTO rollout VALUES(?,?,?,?)',(key,0,0,0))
            cases.append(dict(rollout_uid=key,state_uid=r['state_uid'],controller_name=r['controller']+'_safety',seed=seed))
    con.commit();con.close();write(root/'batches/safety_confirmation/cases.json',cases)
    prior_root,prior_connect=ev.HERE,ev.connect
    try:
        ev.HERE=root;ev.connect=lambda _:sqlite3.connect(dbpath)
        ev.run()
    finally:ev.HERE,ev.connect=prior_root,prior_connect
    metrics=json.loads((root/'mechanism_metrics.json').read_text())
    for scene in ('ring_exchange','toy_give_way'):
        matched=next(r for r in metrics if r['scene']==scene and r['model']=='state_context' and 'switch_accuracy' in r)
        fixed=next(r for r in metrics if r['scene']==scene and r['model']=='eta_only' and 'switch_accuracy' in r)
        assert matched['switch_accuracy']==1. and fixed['switch_accuracy']==0.
    write(HERE/'evaluation_integration_check.json',dict(passed=True,physical_states=16,
        known_perfect_switch=1.,known_fixed_switch=0.,real_rollouts=0,
        fully_unknown_candidate_included=True,output_path=str(root)))
    print('Evaluation integration check passed; no real labels used',flush=True)


if __name__=='__main__':run()
