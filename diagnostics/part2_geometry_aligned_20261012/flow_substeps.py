"""Frozen action-only geometric decomposition for protocol-fixed initial cases."""
import csv,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import jax
import jax.numpy as jnp
import numpy as np

HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parents[2]
POC=RESEARCH/'Basin_C1_flow_field_poc_20261004';HARD=RESEARCH/'Basin_C1_hard_field_20261006'
sys.path.insert(0,str(POC))
from validation_stage2.run_rollouts import make_runtime
from analyze_geometry import writecsv,wrap

def key_for(uid,root):
    token=int(hashlib.sha256(uid.encode()).hexdigest()[:8],16)
    k=jax.random.fold_in(jax.random.PRNGKey(root),token)
    return jax.random.fold_in(jax.random.fold_in(k,0),0)

def nominal(rt,obs,env,h,k):
    s=k/rt.steps
    z=(rt.to_local(h,env)-rt.mean)/rt.scale
    fz=np.asarray(rt.velocity_fn(obs,jnp.asarray(z),jnp.asarray(s))).reshape(-1,2)
    return rt.to_world(rt.scale*fz,env)

def main():
    tasks=json.loads((HARD/'confirm_q16_tasks.json').read_text())['tasks']
    choose={(sc,suf,j) for sc,base in [('toy_give_way',820000),('ring_exchange',920000)]
            for suf in (1,9,17) for j in (0,4,8,12)}
    by={}
    for t in tasks:
        if t['seed']!=0:continue
        key=(t['state']['scenario'],int(t['state']['uid'][-6:])%100, t['eta_index'])
        if key in choose:by[(t['state']['uid'],t['eta_index'],t['chain'])]=t
    cfg=json.loads((HARD/'screening_manifest.json').read_text())
    rows=[];summary=[];rt_cache={}
    for sc,base,old,new,project in [('toy_give_way',820000,'FT','FF',True),
                                     ('ring_exchange',920000,'TT','TF',False)]:
        if sc not in rt_cache:rt_cache[sc]=make_runtime(sc)
        rt=rt_cache[sc]
        for suf in (1,9,17):
            uid=f'hard_confirm_{"toy" if sc=="toy_give_way" else "ring"}_1_{base+suf}'
            for j in (0,4,8,12):
                c=by[(uid,j,old)];env=rt.make_env(c['state']);eta=np.asarray(c['eta'],float)
                key=key_for(uid,cfg['future_root'])
                obs,safety,flow,safe,delta,_=rt.prepare(env,key,eta)
                zero=np.zeros_like(delta)
                hzero,tzero=rt.integrate(env,obs,key,zero,safety,project=project,traced=True)
                heta,teta=rt.integrate(env,obs,key,delta,safety,project=project,traced=True)
                if sc=='toy_give_way':
                    aold=safety.project(hzero+delta)[0];anew=heta
                    e=(env.goals-env.positions)/np.linalg.norm(env.goals-env.positions,axis=1,keepdims=True)
                    wgoal=(e/len(e)).ravel();rel=env.positions[0]-env.positions[1]
                    wclear=np.r_[2*rel,-2*rel];w=wgoal
                    ambiguous=0
                else:
                    aold=safety.project(safe+delta)[0];anew=safety.project(heta)[0]
                    p=env.positions;r=np.linalg.norm(p,axis=1)
                    tangent=np.stack((-p[:,1],p[:,0]),axis=1)/r[:,None]
                    gap=wrap(np.arctan2(env.goals[:,1],env.goals[:,0])-np.arctan2(p[:,1],p[:,0]))
                    w=(np.sign(gap)[:,None]*tangent/r[:,None]/len(p)).ravel()
                    wgoal=None;wclear=None;ambiguous=int(np.sum(np.pi-np.abs(gap)<.05))
                cache_error={}
                for ch,a in ((old,aold),(new,anew)):
                    task=by[(uid,j,ch)]
                    with np.load(HARD/'results'/f"{task['id']}.npz") as z:cached=z['actions'][0]
                    cache_error[ch]=float(np.linalg.norm(a-cached))
                assert max(cache_error.values())<1e-7,(sc,uid,j,cache_error)
                cum_delta=cum_fb=cum_proj=0.
                for k in range(rt.steps):
                    ds=1/rt.steps
                    z0=tzero['path'][k];z1=teta['path'][k]
                    u0=nominal(rt,obs,env,z0,k);u1=nominal(rt,obs,env,z1,k)
                    direct=ds*delta;feedback=ds*(u1-u0)
                    actual=(teta['path'][k+1]-teta['path'][k])-(tzero['path'][k+1]-tzero['path'][k])
                    projection=actual-direct-feedback
                    residual=float(np.linalg.norm(actual-direct-feedback-projection))
                    cum_delta+=float(np.dot(w,direct.ravel()))
                    cum_fb+=float(np.dot(w,feedback.ravel()))
                    cum_proj+=float(np.dot(w,projection.ravel()))
                    row=dict(scene=sc,uid=uid,eta_index=j,seed=0,substep=k+1,
                        field_projected=int(project),ambiguous_agents=ambiguous,
                        delta_norm=float(np.linalg.norm(delta)),
                        direct_directional_increment=float(np.dot(w,direct.ravel())),
                        feedback_directional_increment=float(np.dot(w,feedback.ravel())),
                        field_projection_directional_increment=float(np.dot(w,projection.ravel())),
                        cumulative_direct=cum_delta,cumulative_feedback=cum_fb,cumulative_projection=cum_proj,
                        path_directional_difference=float(np.dot(w,(teta['path'][k+1]-tzero['path'][k+1]).ravel())),
                        path_difference_norm=float(np.linalg.norm(teta['path'][k+1]-tzero['path'][k+1])),
                        step_decomposition_residual=residual,
                        zero_field_projection_norm=tzero['steps'][k]['field_projection_norm'],
                        eta_field_projection_norm=teta['steps'][k]['field_projection_norm'])
                    if sc=='toy_give_way':
                        row['clearance_path_difference']=float(np.dot(wclear,(teta['path'][k+1]-tzero['path'][k+1]).ravel()))
                    rows.append(row)
                terminal_effect=float(np.dot(w,(anew-heta).ravel())) if sc=='ring_exchange' else 0.
                summary.append(dict(scene=sc,uid=uid,eta_index=j,seed=0,old=old,new=new,
                    ambiguous_agents=ambiguous,field_path_directional_difference=float(np.dot(w,(heta-hzero).ravel())),
                    field_path_norm=float(np.linalg.norm(heta-hzero)),
                    direct_directional_total=cum_delta,feedback_directional_total=cum_fb,
                    field_projection_directional_total=cum_proj,
                    terminal_projection_on_eta_endpoint=terminal_effect,
                    final_pair_directional_difference=float(np.dot(w,(anew-aold).ravel())),
                    old_cached_action_error=cache_error[old],new_cached_action_error=cache_error[new],
                    new_action_max_violation=safety.certificate(anew)['max_violation']))
    writecsv(HERE/'flow_substep_decomposition.csv',rows)
    writecsv(HERE/'flow_substep_summary.csv',summary)
    print('action-only cases',len(summary),'substep rows',len(rows),'max cache error',
          max(max(r['old_cached_action_error'],r['new_cached_action_error']) for r in summary))

if __name__=='__main__':main()
