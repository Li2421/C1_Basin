"""Read-only SBMA provenance, trajectory, terminal and retry verification."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from diagnostics.success_basin_multimodality.setup import HERE,ROOT,SYSROOT,sha,write
from diagnostics.success_basin_multimodality.exact_projector import (project_velocity_with_retry,
    tighter_identical_socp,identical_nonlinear_polish)
from diagnostics.astra_true_q_audit.audit import restore
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.cbf import CBFConfig,barrier_constraints

def main():
    p=json.loads((HERE/'protocol.json').read_text());cfg=Config(**p['environment']);cbf=CBFConfig();cat={x['state_id']:x for x in p['primary_states']}
    assert sha(SYSROOT/'single_integrator/cbf.py')==p['solver']['current_source_sha256']
    assert sha(SYSROOT/'single_integrator/environment.py')==p['source_hashes']['environment']
    assert sha(ROOT/'diagnostics/cl_fhcb/closed_loop.py')==p['source_hashes']['corrector']
    assert sha(p['checkpoint'])==p['source_hashes']['flow_checkpoint']
    prior=json.loads((HERE.parent/'success_basin_geometry/manifest.json').read_text())
    for item in prior['artifacts']:
        path=HERE.parent/'success_basin_geometry'/item['path'];assert sha(path)==item['sha256'],path
    rows=[];stages={};maxinteg=maxg=maxspeed=max_projection_recompute_difference=0.;counts=Counter();raw_attempts=0;rawsteps=0
    for mf in sorted((HERE/'raw').glob('*/manifest.json')):
        m=json.loads(mf.read_text());stages[m['stage']]={'attempts':len(m['records']),'steps':sum(r['steps'] for r in m['records'])};rows+=m['records']
    for r in rows:
        path=HERE/r['file'];assert sha(path)==r['sha256'];raw_attempts+=1;rawsteps+=r['steps'];counts[r['outcome']]+=1
        with np.load(path) as d:
            n=r['steps'];assert len(d['event'])==n
            if n:
                maxinteg=max(maxinteg,float(np.max(abs(d['positions_after']-d['positions_before']-.05*d['u_exec']))))
                maxspeed=max(maxspeed,float(np.linalg.norm(d['u_exec'],axis=-1).max()))
                goals=np.asarray([[1.09,0.],[-1.09,0.]])
                bg=(goals[None]-d['positions_before']).astype(np.float32).astype(np.float64)
                br=(d['positions_before']-d['positions_before'][:,::-1]).astype(np.float32).astype(np.float64)
                for base in [bg,br]:base*=np.minimum(1.,.5/np.maximum(np.linalg.norm(base,axis=-1,keepdims=True),1e-30))
                eta=np.asarray(r['eta']);expected=eta[0]*bg+eta[1]*d['u_safe']+eta[2]*br
                maxg=max(maxg,float(np.max(abs(expected-d['g']))))
            if r['outcome'] is None:
                assert r['execution_error'] and (not n or np.all(d['event']=='running'))
            else:
                assert n and str(d['event'][-1])==r['outcome'] and np.all(d['event'][:-1]=='running')
                if r['outcome']=='success':assert np.all(np.linalg.norm(goals-d['positions_after'][-1],axis=-1)<=cfg.goal_tolerance)
                if r['outcome']=='timeout':assert cat[r['state_id']]['start_step']+n==cfg.max_steps
    assert maxinteg<1e-12 and maxg<1e-12 and maxspeed<=cfg.max_speed+1e-9
    # Fully replay a deterministic set spanning state, terminal outcome, endpoints, center and numerical retries.
    chosen={};targets={tuple([.75,-.5,.75]),tuple([1.,0.,.25]),tuple([1.25,.5,0.])}
    for r in rows:
        group=(r['state_id'],r['outcome'],tuple(np.round(r['eta'],12)) if tuple(np.round(r['eta'],12)) in targets else None)
        if r['outcome'] is not None and group not in chosen:chosen[group]=r
    projection_checks=0
    for r in chosen.values():
        entry=cat[r['state_id']]
        with np.load(HERE.parent/'success_basin_geometry'/entry['state_file']) as s:env=restore(dict(s),cfg)
        with np.load(HERE/r['file']) as d:
            for j,u in enumerate(d['u_exec']):
                assert np.allclose(env.positions,d['positions_before'][j],atol=1e-12,rtol=0)
                if j in {0,len(d['event'])//2,len(d['event'])-1}:
                    A,b,_=barrier_constraints(env.snapshot(),cbf)
                    us,_,_,_=project_velocity_with_retry(d['u_flow'][j],A,b,cfg.max_speed,cbf)
                    ue,_,_,_=project_velocity_with_retry(d['w'][j],A,b,cfg.max_speed,cbf)
                    de1=float(np.max(abs(us-d['u_safe'][j])));de2=float(np.max(abs(ue-u)))
                    max_projection_recompute_difference=max(max_projection_recompute_difference,de1,de2)
                    if de1>5e-6 or de2>5e-6:raise AssertionError((r['file'],j,de1,de2))
                    projection_checks+=2
                _,_,done,info=env.step(u);assert info['termination']==str(d['event'][j]);assert np.allclose(env.positions,d['positions_after'][j],atol=1e-12,rtol=0)
    # Independently recompute every accepted fallback action and retain its KKT certificate.
    retries=[]
    for r in rows:
        if r['outcome'] is None:continue
        with np.load(HERE/r['file']) as d:
            if 'first_retry' not in d:continue
            entry=cat[r['state_id']]
            for which,targetname,outname,flagname in [('first','u_flow','u_safe','first_retry'),('second','w','u_exec','second_retry')]:
                for j in np.flatnonzero(d[flagname]):
                    env=GiveWayEnv(cfg);snap=env.snapshot();snap['positions']=d['positions_before'][j]
                    A,b,_=barrier_constraints(snap,cbf);u,status,retried,diag=project_velocity_with_retry(d[targetname][j],A,b,cfg.max_speed,cbf)
                    stored=np.asarray(d[outname][j]).reshape(4);target=np.asarray(d[targetname][j]).reshape(4)
                    retry_difference=float(np.max(abs(np.asarray(u).reshape(4)-stored)))
                    linear=A@stored-b;speed_res=.25-np.sum(stored.reshape(2,2)**2,axis=1);active_rows=[];active_res=[]
                    for row,value in zip(A,linear):
                        if value<=1e-7:active_rows.append(row);active_res.append(value)
                    for agent,value in enumerate(speed_res):
                        if value<=1e-7:
                            row=np.zeros(4);row[2*agent:2*agent+2]=-2*stored.reshape(2,2)[agent]
                            active_rows.append(row);active_res.append(value)
                    multipliers,_=nnls(np.asarray(active_rows).T,stored-target,maxiter=5000)
                    stationarity=float(np.linalg.norm(stored-target-np.asarray(active_rows).T@multipliers,ord=np.inf))
                    complementarity=float(np.max(abs(multipliers*np.asarray(active_res))))
                    direct={'min_linear_residual':float(linear.min()),
                      'max_speed_excess':float(np.linalg.norm(stored.reshape(2,2),axis=1).max()-.5),
                      'stationarity_residual':stationarity,'complementarity_residual':complementarity}
                    assert direct['min_linear_residual']>=-cbf.feasibility_tol and direct['max_speed_excess']<=cbf.speed_tol
                    direct['simple_active_set_certificate_pass']=(direct['min_linear_residual']>=-cbf.feasibility_tol and
                      direct['max_speed_excess']<=cbf.speed_tol and max(stationarity,complementarity)<=cbf.optimality_tol)
                    stored_status=str(d[f'{which}_status'][j])
                    forced=(tighter_identical_socp if stored_status=='solved_identical_socp_tighter_retry' else identical_nonlinear_polish)
                    forced_u,forced_status,forced_diag=forced(target,A,b,cfg.max_speed,cbf)
                    forced_difference=float(np.max(abs(np.asarray(forced_u).reshape(4)-stored)))
                    assert forced_difference<=3e-8
                    retries.append({'file':r['file'],'step_index':int(j),'projection':which,
                      'stored_status':stored_status,'recomputed_status':status,
                      'recomputed_retry_triggered':retried,'recomputed_diagnostic':diag,
                      'recompute_action_max_difference':retry_difference,
                      'forced_original_fallback_status':forced_status,'forced_original_fallback_action_difference':forced_difference,
                      'forced_original_fallback_certificate':forced_diag,'direct_stored_action_certificate':direct})
    audit=json.loads((HERE/'projection_solver_audit.json').read_text());assert audit['classification_counts']=={'NUMERICAL_SOLVER_FAILURE':754}
    # Independent final-table checks.
    path=json.loads((HERE/'path_connectivity.json').read_text());assert path['result']=='SUCCESS_PATH_FOUND'
    assert all(c['counts']=={'success':32,'deadlock':0,'timeout':0,'collision':0} for cs in path['per_state'].values() for c in cs)
    valid=json.loads((HERE/'fresh_component_validation.json').read_text());assert all(x['counts']['success']==32 for x in valid['known_success_eta'])
    out={'raw_attempts':raw_attempts,'raw_physical_steps':rawsteps,'raw_outcome_counts':{str(k):v for k,v in counts.items()},
      'stage_totals':stages,'maximum_integration_residual':maxinteg,'maximum_corrector_residual':maxg,
      'maximum_executed_speed':maxspeed,'representative_full_replays':len(chosen),'projection_recomputations':projection_checks,
      'maximum_projection_recompute_difference':max_projection_recompute_difference,
      'accepted_identical_problem_retry_actions':len(retries),'retry_certificates':retries,
      'old_754_unknown_classification_verified':True,'old_artifacts_unchanged':True,
      'path_table_independently_checked':'PASS','known_eta_validation_independently_checked':'PASS'}
    write('verification.json',out);print(json.dumps({k:v for k,v in out.items() if k!='retry_certificates'},indent=2))

if __name__=='__main__':main()
