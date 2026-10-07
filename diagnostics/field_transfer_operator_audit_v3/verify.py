"""Independent artifact consistency and numerical checks, without new action calls."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pandas as pd
from cache_analysis import HERE,CACHE,PARENT,read,save,verify,CHAINS

def main():
    checks,_=verify()
    for name in ['cache_analysis.py','local_probes.py','targeted_checks.py','state_chart_analysis.py','trajectory_linearization.py','plot_audit.py','inspect_runtime.py']:
        ast.parse((HERE/name).read_text())
        checks['syntax:'+name]=True
    # Independent counts from interval labels, not the generated group table.
    expected={('toy_give_way','both_success'):19,('toy_give_way','loss'):101,('toy_give_way','rescue'):16,('toy_give_way','both_failure'):248,
              ('ring_exchange','both_success'):29,('ring_exchange','loss'):1,('ring_exchange','rescue'):46,('ring_exchange','both_failure'):308}
    counter={key:0 for key in expected}
    for row in read(PARENT/'paired_states.json'):
        for j in range(16):
            labs=[]
            for ch in ['TT','FF']:
                lo,hi=row['cells'][ch]['Q_distribution'][j]
                assert lo>=15/16 or hi<15/16
                labs.append(lo>=15/16)
            a,b=labs;g='both_success' if a and b else 'loss' if a else 'rescue' if b else 'both_failure'
            counter[row['scene'],g]+=1
    assert counter==expected
    checks['independent_B15_counts']=True
    quality=pd.read_csv(HERE/'all_context_quality.csv.gz')
    assert len(quality)==21504*4
    assert quality[quality.chain=='FF'].reliable.sum()==21397
    assert sum(read(HERE/'geometry.json')[s+':FF']['reliable'] for s in ['toy_give_way','ring_exchange'])==6129+6124
    checks['quality_counts']=True
    # Recalculate a finite candidate displacement directly from two archived outputs.
    ini=pd.read_csv(HERE/'initial_context_metrics.csv.gz')
    etas=np.array(read(PARENT/'screening_manifest.json')['eta'])
    for sc in ['toy_give_way','ring_exchange']:
        r=ini[(ini.scene==sc)&(ini.chain=='FF')&ini.reliable].iloc[0]
        z=np.load(CACHE/'results'/f'{r.task_id}.npz');j=int(r.eta_index);d=z['context0_basis'].shape[0]
        distances=np.linalg.norm(etas-etas[j],axis=1);distances[j]=np.inf;k=distances.argmin()
        jj=z[f'context0_center{j}_J_eta_exec'][3,2]
        stride=1+6*d+18
        change=z['context0_exec_all'][3,1+stride*k]-z['context0_exec_all'][3,1+stride*j]
        e=np.linalg.norm(change-jj@(etas[k]-etas[j]))
        np.testing.assert_allclose(e,r.nearest_error,rtol=1e-12,atol=1e-14)
        checks['candidate_error_direct:'+sc]=True
    rec=read(HERE/'recurrence_checks.json')
    assert len(rec)==48
    assert max(r['raw_AD_FD_relative_error'] for r in rec)<.004
    assert max(r['FF_KKT_FD_relative_error'] for r in rec)<.007
    assert not any(r['FF_weak_steps'] for r in rec)
    assert max(r['error'] for r in read(HERE/'probe_replay_checks.json'))==0
    checks['independent_variational_validation']=True
    rays=read(HERE/'scale_probe_rows.json');assert len(rays)==48*3*6*2
    assert read(HERE/'probe_numerical_failures.json')==[]
    paths=read(HERE/'path_integral_checks.json');assert len(paths)==8
    assert max(r['quadrature_errors']['33']['relative_error'] for r in paths)<.011
    assert not any(r['weak_projection_steps'] for r in paths)
    checks['finite_scale_and_integral_validation']=True
    trajectory=read(HERE/'trajectory_linearization.json')
    assert trajectory['npz_hashes_checked']==3072
    assert sum(v['total'] for v in trajectory['summary'].values())==9216*4
    checks['all_trajectory_probe_hashes_and_counts']=True
    ex=read(HERE/'validated_counterexample_operators.json')
    refl=next(r for r in ex if r['purpose']=='reflection')
    assert refl['determinant']<0 and refl['AD_KKT_determinant']<0 and refl['AD_KKT_relative_error']<2e-6
    assert len(read(HERE/'other_scenario_probes.json'))==4
    checks['counterexample_validation']=True
    # No controller/plant step is present in any new scientific probe source.
    for filename in ['local_probes.py','targeted_checks.py']:
        tree=ast.parse((HERE/filename).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        assert not any(isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in calls)
    checks['no_rollout_execution_in_probes']=True
    files={}
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and '__pycache__' not in str(p) and p.name not in ['verification.json','artifact_inventory.json']:
            files[str(p.relative_to(HERE))]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    save('artifact_inventory.json',files)
    save('verification.json',dict(passed=all(checks.values()),checks=checks,new_full_rollouts=0,
        source_cache_mutated=False,limitations='Hash coverage is explicit in provenance.json; action associations are exploratory.'))
    print(json.dumps(dict(passed=True,checks=len(checks),new_full_rollouts=0)))

if __name__=='__main__':main()
