"""Verify selected counterexamples and integrate the local derivative along a correction ray."""
import json
import numpy as np
import pandas as pd
from local_probes import (HERE,PARENT,CACHE,FIELD,read,save,stats,rel,make_runtime,
                         recurrence,jax,jnp,projection_derivative)

def main():
    cfg=read(PARENT/'screening_manifest.json');etas=np.array(cfg['eta'])
    states={s['uid']:s for s in read(PARENT/'confirmation_manifest.json')['states']}
    tasks=read(CACHE/'tasks.json')['tasks'];tm={t['id']:t for t in tasks}
    df=pd.read_csv(HERE/'initial_context_metrics.csv.gz')
    ff=df[(df.chain=='FF') & df.reliable]
    selection=[]
    for sc in ['toy_give_way','ring_exchange']:
        g=ff[ff.scene==sc]
        selection.append(('max_amplification',g.loc[g.sigma_max.idxmax()].to_dict()))
        if len(g[g.determinant_identified & g.negative_determinant]):
            q=g[g.determinant_identified & g.negative_determinant]
            selection.append(('reflection',q.loc[q.sigma_min.idxmax()].to_dict()))
        selection.append(('rank_collapse',g.loc[g.sigma_min.idxmin()].to_dict()))
    records=[]
    for sc in ['toy_give_way','ring_exchange']:
        rt=make_runtime(sc);grad=jax.jit(jax.jacrev(rt.velocity_fn,argnums=1))
        for purpose,r in [(p,r) for p,r in selection if r['scene']==sc]:
            state=states[r['state_uid']];env=rt.make_env(state)
            z=np.load(CACHE/'results'/f"{r['task_id']}.npz")
            key=jnp.asarray(z['context0_noise_key'],dtype=jnp.uint32);j=int(r['eta_index'])
            obs,safety,flow,safe,_,_=rt.prepare(env,key,np.zeros(3))
            b=z['context0_basis'];delta=(b@etas[j]).reshape(safe.shape)
            raw,ff_action,rr=recurrence(rt,env,obs,key,delta,safety,grad)
            m=z[f'context0_center{j}_M_exec'][3,2];jad=rr[1]['M'];u,s,vh=np.linalg.svd(m)
            records.append(dict(purpose=purpose,scene=sc,state_uid=state['uid'],task_id=r['task_id'],seed=int(r['seed']),eta_index=j,
                group=r['group'],eta=etas[j].tolist(),M=m.tolist(),J_eta=(m@b).tolist(),B=b.tolist(),U=u.tolist(),singular_values=s.tolist(),Vh=vh.tolist(),
                determinant=float(np.linalg.det(m)),AD_KKT_determinant=float(np.linalg.det(jad)),
                AD_KKT_relative_error=rel(jad,m),weak_steps=sum(q['weak'] for q in rr[1]['steps']),
                epsilon_change=r['epsilon_relative_change'],chainrule_error=r['chain_rule_relative_error']))
    save('validated_counterexample_operators.json',records)
    # Predetermined state offsets and candidate; no outcome selection.
    control=read(CACHE/'state_dependence_control/manifest.json')
    integral=[]
    for sc in ['toy_give_way','ring_exchange']:
        rt=make_runtime(sc);grad=jax.jit(jax.jacrev(rt.velocity_fn,argnums=1))
        selected=[s for s in states.values() if s['scenario']==sc]
        for ix in [0,7,15,23]:
            state=selected[ix];env=rt.make_env(state);key=jnp.asarray(control['actual_keys'][sc],dtype=jnp.uint32)
            obs,safety,flow,safe,_,_=rt.prepare(env,key,np.zeros(3))
            b=np.stack(rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values,axis=-1).reshape(-1,3)
            j=4;delta=(b@etas[j]).reshape(safe.shape);samples=[];outputs=[];weak=0
            for alpha in np.linspace(0,1,33):
                raw,value,rr=recurrence(rt,env,obs,key,alpha*delta,safety,grad)
                samples.append(rr[1]['M']@delta.ravel());outputs.append(value.ravel())
                weak+=sum(q['weak'] for q in rr[1]['steps'])
            samples=np.array(samples);actual=outputs[-1]-outputs[0];norm=np.linalg.norm(actual)
            errs={}
            for stride in [4,2,1]:
                prediction=np.trapezoid(samples[::stride],dx=stride/32,axis=0)
                errs[str(32//stride+1)]=dict(error=np.linalg.norm(actual-prediction),relative_error=np.linalg.norm(actual-prediction)/max(norm,1e-8))
            integral.append(dict(scene=sc,state_uid=state['uid'],eta_index=j,actual=actual,
                zero_J_prediction=samples[0],single_relative_error=np.linalg.norm(actual-samples[0])/max(norm,1e-8),
                quadrature_errors=errs,weak_projection_steps=weak))
            print('integrated local derivative',state['uid'],errs,flush=True)
    save('path_integral_checks.json',integral)
    # Genuine failure of an epsilon-only stability rule at a projection boundary.
    boundary=[]
    for sc in ['toy_give_way','ring_exchange']:
        ids=[s['uid'] for s in states.values() if s['scenario']==sc]
        for uid in ids:
            q=read(CACHE/'state_dependence_control'/f'{uid}.json')['diagnostics']['chains']['TT']
            if not q['fd_stable']:
                boundary.append(dict(scene=sc,state_uid=uid,chain='TT',center_eta=[0,0,0],
                    epsilon_change=q['epsilon_relative_change'],one_sided_mismatch=q['one_sided_mismatch'],
                    singular_values=q['M_exec']['singular_values']))
                break
    save('boundary_counterexamples.json',boundary)
    unstable=[]
    aq=pd.read_csv(HERE/'all_context_quality.csv.gz')
    for sc in ['toy_give_way','ring_exchange']:
        g=aq[(aq.scene==sc)&(aq.chain=='FF')&(~aq.reliable)&aq.epsilon_relative_change.notna()]
        for _,row in g.sort_values('one_sided_mismatch',ascending=False).head(3).iterrows():
            unstable.append(row.where(pd.notna(row),None).to_dict())
    save('unstable_FF_counterexamples.json',unstable)
    print('targeted verification complete',flush=True)

if __name__=='__main__':main()
