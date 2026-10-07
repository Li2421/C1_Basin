"""Read-only SBGA trace, safety, terminal-event and provenance validation."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from diagnostics.success_basin_geometry.setup import HERE,ROOT,sha,write
from diagnostics.success_basin_geometry.analyze import read,records
from diagnostics.astra_true_q_audit.audit import restore
from single_integrator.environment import Config
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity

def main():
    p=read('protocol.json');cat={s['state_id']:s for s in p['state_catalog']};cfg=Config(**p['environment']);cbf=CBFConfig()
    old=json.loads((ROOT/'diagnostics/true_q_geometry/manifest.json').read_text())
    for group in ['frozen_sources_and_specs','study_code','raw_indexes','deliverables_and_figures']:
        for r in old[group]:assert sha(ROOT/r['path'])==r['sha256']
    rows=records();count=Counter();maxintegration=0.;maxg=0.;maxspeed=0.;replayed=0;projected=0
    chosen=set()
    # Replays cover each state/outcome plus each independently validated cell.
    validationfiles={r['file'] for m in (HERE/'raw').glob('validation*/manifest.json') for r in json.loads(m.read_text())['records']}
    for index,r in enumerate(rows):
        path=HERE/r['file'];assert sha(path)==r['sha256']
        with np.load(path) as data:d=dict(data)
        n=r['steps'];assert len(d['event'])==n
        count[str(r['outcome'])]+=1
        if n:
            maxintegration=max(maxintegration,float(np.max(abs(d['positions_after']-d['positions_before']-.05*d['u_exec']))))
            maxspeed=max(maxspeed,float(np.linalg.norm(d['u_exec'],axis=-1).max()))
            goals=np.array([[1.09,0],[-1.09,0]])
            # Exact observation semantics include float32 conversion before the basis bounds.
            bg=(goals[None]-d['positions_before']).astype(np.float32).astype(np.float64)
            br=(d['positions_before']-d['positions_before'][:,::-1]).astype(np.float32).astype(np.float64)
            for b in [bg,br]:b*=np.minimum(1.,.5/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-30))
            phi=np.array(r['phi']);expect=phi[0]*bg+phi[1]*d['u_safe']+phi[2]*br
            maxg=max(maxg,float(np.max(abs(expect-d['g']))))
        if r['outcome'] is not None:
            assert n and str(d['event'][-1])==r['outcome'];assert np.all(d['event'][:-1]=='running')
            if r['outcome']=='success':assert np.all(np.linalg.norm(goals-d['positions_after'][-1],axis=-1)<=cfg.goal_tolerance)
            if r['outcome']=='timeout':assert cat[r['state_id']]['start_step']+n==850
        else:assert r['execution_error'] and (not n or np.all(d['event']=='running'))
        group=(r['state_id'],tuple(r['phi']) if r['file'] in validationfiles else None,r['outcome'])
        if group not in chosen:
            chosen.add(group)
            with np.load(HERE/cat[r['state_id']]['state_file']) as s:env=restore(dict(s),cfg)
            for j,u in enumerate(d['u_exec']):
                if j in {0,n//2,n-1}:
                    A,b,_=barrier_constraints(env.snapshot(),cbf)
                    us,_=project_velocity(d['u_flow'][j],A,b,.5,cbf);ue,_=project_velocity(d['w'][j],A,b,.5,cbf)
                    assert np.allclose(us,d['u_safe'][j],atol=1e-9,rtol=0)
                    assert np.allclose(ue,u,atol=1e-9,rtol=0);projected+=1
                _,_,done,info=env.step(u)
                assert info['termination']==str(d['event'][j])
                assert np.allclose(env.positions,d['positions_after'][j],atol=1e-12,rtol=0)
            replayed+=1
    assert maxintegration<1e-12 and maxg<1e-12 and maxspeed<=.5+1e-9
    write('verification.json',{'indexed_trajectories':len(rows),'counts':dict(count),'all_trace_kinematics_and_corrector':'PASS',
        'maximum_integration_residual':maxintegration,'maximum_corrector_residual':maxg,'maximum_agent_speed':maxspeed,
        'fully_replayed_representatives':replayed,'hard_projection_recomputations':projected,
        'old_sources_and_results_unchanged':True,'final_outcomes_never_assigned_to_solver_failures':True})
    print(read('verification.json'))

if __name__=='__main__':main()
