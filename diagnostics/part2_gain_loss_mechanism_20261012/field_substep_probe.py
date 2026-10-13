"""Four preregistered t0, eta[4], seed0 action-only Flow traces; no rollout."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import jax
import numpy as np

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parents[2]
POC=RESEARCH/'Basin_C1_flow_field_poc_20261004'
HARD=RESEARCH/'Basin_C1_hard_field_20261006'
sys.path.insert(0,str(POC))
from validation_stage2.run_rollouts import make_runtime
from validation_stage2.factorial_controller import action

CASES=(('toy_give_way',820001,'FT','FF',True),
       ('toy_give_way',820002,'FT','FF',True),
       ('ring_exchange',920001,'TT','TF',False),
       ('ring_exchange',920002,'TT','TF',False))

def key_for(uid,seed,step,root):
    token=int(hashlib.sha256(uid.encode()).hexdigest()[:8],16)
    k=jax.random.fold_in(jax.random.PRNGKey(root),token)
    return jax.random.fold_in(jax.random.fold_in(k,seed),step)

def main():
    tasks=json.loads((HARD/'confirm_q16_tasks.json').read_text())['tasks']
    cases={}; all_tasks={}
    for t in tasks:
        if t['eta_index']==4 and t['seed']==0:
            all_tasks[(t['state']['uid'],t['chain'])]=t
            if t['chain']=='TT':cases[t['state']['uid']]=t
    cfg=json.loads((HARD/'screening_manifest.json').read_text())
    rows=[];summary=[]
    for scene,suffix,old,new,project in CASES:
        uid=next(u for u in cases if u.endswith(str(suffix)))
        c=cases[uid];rt=make_runtime(scene);env=rt.make_env(c['state'])
        eta=np.asarray(c['eta'],float);key=key_for(uid,0,0,cfg['future_root'])
        obs,safety,flow,safe,delta,_=rt.prepare(env,key,eta)
        zero=np.zeros_like(delta)
        h0,t0=rt.integrate(env,obs,key,zero,safety,project=project,traced=True)
        h1,t1=rt.integrate(env,obs,key,delta,safety,project=project,traced=True)
        aold,_,_,_=action(rt,env,key,eta,old)
        anew,_,_,_=action(rt,env,key,eta,new)
        cached_old=np.load(HARD/'results'/f"{all_tasks[(uid,old)]['id']}.npz")['actions'][0]
        cached_new=np.load(HARD/'results'/f"{all_tasks[(uid,new)]['id']}.npz")['actions'][0]
        cache_old_error=float(np.linalg.norm(aold-cached_old))
        cache_new_error=float(np.linalg.norm(anew-cached_new))
        assert max(cache_old_error,cache_new_error)<1e-7,(scene,uid,cache_old_error,cache_new_error)
        reconstructed=h1 if project else safety.project(h1)[0]
        assert np.linalg.norm(anew-reconstructed)<1e-8,(scene,uid,np.linalg.norm(anew-reconstructed))
        goal=env.goals-env.positions
        goal/=np.maximum(np.linalg.norm(goal,axis=1,keepdims=True),1e-12)
        r=env.positions/np.maximum(np.linalg.norm(env.positions,axis=1,keepdims=True),1e-12)
        tangent=np.stack((-r[:,1],r[:,0]),axis=1)
        dnorm=np.linalg.norm(delta)
        for k,(v0,v1) in enumerate(zip(t0['path'],t1['path'])):
            diff=v1-v0
            row=dict(scene=scene,uid=uid,eta_index=4,seed=0,field_projected=int(project),
                substep=k,delta_norm=dnorm,path_difference_norm=float(np.linalg.norm(diff)),
                path_difference_over_delta=float(np.linalg.norm(diff)/max(dnorm,1e-12)),
                linear_no_feedback_reference=float(k/rt.steps),
                goal_signed_difference=float(np.sum(diff*goal)),
                tangential_difference_norm=float(np.linalg.norm(np.sum(diff*tangent,axis=1))),
                old_new_action_norm=float(np.linalg.norm(anew-aold)))
            if k:
                row['zero_field_projection_norm']=t0['steps'][k-1]['field_projection_norm']
                row['eta_field_projection_norm']=t1['steps'][k-1]['field_projection_norm']
                row['zero_solver_status']=t0['steps'][k-1]['solver_status']
                row['eta_solver_status']=t1['steps'][k-1]['solver_status']
            rows.append(row)
        summary.append(dict(scene=scene,uid=uid,eta_index=4,seed=0,old=old,new=new,
            delta_norm=float(dnorm),final_path_difference_norm=float(np.linalg.norm(h1-h0)),
            field_path_ratio=float(np.linalg.norm(h1-h0)/max(dnorm,1e-12)),
            final_action_difference_norm=float(np.linalg.norm(anew-aold)),
            new_action_reconstruction_error=float(np.linalg.norm(anew-reconstructed)),
            cached_old_action_error=cache_old_error,cached_new_action_error=cache_new_error,
            final_new_cert=safety.certificate(anew),
            new_projection_active_steps=sum(s['field_projection_norm']>1e-7 for s in t1['steps'])))
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/'field_substep_probe.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    (HERE/'field_substep_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,3.9))
    for x in summary:
        rr=[r for r in rows if r['uid']==x['uid']]
        label=f"{x['scene'].split('_')[0]} {x['uid'][-6:]}"
        ax.plot([r['substep'] for r in rr],[r['path_difference_over_delta'] for r in rr],
                marker='o',markersize=2.5,label=label)
    ax.plot(range(11),np.arange(11)/10,color='0.55',linestyle='--',label='linear, no feedback')
    ax.set(xlabel='Flow substep',ylabel=r'$\|h_k(\delta)-h_k(0)\|/\|\delta\|$',
           title='Fixed initial state, eta[4], seed 0; action-only trace')
    ax.legend(fontsize=8,ncol=2);fig.tight_layout()
    fig.savefig(HERE/'figures/field_substep_eta4_seed0.svg');plt.close(fig)
    for x in summary:print(x['scene'],x['uid'],'path_ratio',round(x['field_path_ratio'],3),
                          'action_diff',round(x['final_action_difference_norm'],3),
                          'active_steps',x['new_projection_active_steps'])

if __name__=='__main__':main()
