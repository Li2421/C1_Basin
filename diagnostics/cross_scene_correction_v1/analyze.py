"""Fit and seal all prospective decisions before any extrapolation main result."""
from common import *
from stages import read_result
from models import select,crossfit,calibration
import numpy as np


def tables():
    scenes=[s for s in read(HERE/'SCENES.json')if s['family']!='extrapolation']
    selected=read(HERE/'SELECTED_CONTROLLERS.json');cases=read(HERE/'main_CASES.json')
    by_scene={};outcomes=[]
    action_ids={etahex(a['eta_a']):i for i,a in enumerate(actions())}
    for s in scenes:
        ctls=[v for v in selected[s['name']].values()if v is not None]
        if not ctls:
            raise RuntimeError('NO_QUALIFIED_CONTROLLER: '+s['name'])
        by_scene[s['name']]=dict(controllers=[c['controller']for c in ctls],state_indices=[q['index']for q in s['states']],
            y=np.full((len(ctls),len(s['states']),49),np.nan),unknown=np.full((len(ctls),len(s['states']),49),np.nan),
            collision=np.full((len(ctls),len(s['states']),49),np.nan))
    for c in cases:
        r=read_result('main',c);name=c['state']['scene'];row=by_scene[name]
        i=row['controllers'].index(c['controller_payload']['nominal']['controller']);j=row['state_indices'].index(c['state']['index']);a=action_ids[etahex(c['eta_a'])]
        for key,field in [('y','success'),('unknown','numerical_failure'),('collision','collision')]:
            assert np.isnan(row[key][i,j,a]);row[key][i,j,a]=r[field]
        outcomes.append(dict(scene=name,controller=row['controllers'][i],state=j,action=a,rollout_uid=c['rollout_uid'],
            success=r['success'],unknown=r['numerical_failure'],collision=r['collision'],timeout=r['timeout'],
            collision_types=r.get('collision_types'),episode_length=r['episode_length'],elapsed_seconds=r['elapsed_seconds']))
    for row in by_scene.values():
        for key in ('y','unknown','collision'):assert np.isfinite(row[key]).all()
        row['type_equal_candidate_success_lower']=row['y'].mean((0,1))
        row['type_equal_candidate_success_upper']=np.minimum(1.,row['y']+row['unknown']).mean((0,1))
    write(HERE/'MAIN_OUTCOME_ROWS.json',outcomes)
    write(HERE/'LABEL_TABLES.json',by_scene)
    return scenes,by_scene


def main():
    assert not (HERE/'extrapolation_CASES.json').exists(), 'No extrapolation main outcomes before freeze'
    plan_proof=HERE/'MAIN_PLAN_COVERAGE_AUDIT.json'
    if not plan_proof.exists():
        from audit_main_plan import main as audit_main_plan
        audit_main_plan()
    proof=read(plan_proof);assert proof['passed']
    for name,expected in proof['files'].items():assert sha(name)==expected,('DECLARED_MAIN_PLAN_CHANGED',name)
    cpath=HERE/'C_CRITERION_CLARIFICATION.json';csha=sha(cpath)if cpath.exists()else None
    crule=read(cpath)['decision_rule']if cpath.exists()else 'mean_difference_le_0.05'
    scenes,data=tables();f=np.asarray([s['features']for s in scenes]);names=[s['name']for s in scenes]
    q=np.asarray([data[n]['y'].mean((0,1))for n in names]);etas=np.asarray([a['eta_a']for a in actions()])
    loso={}
    for i,s in enumerate(scenes):
        train=[j for j in range(len(scenes))if j!=i]
        choices=select(f[train],q[train],f[i],etas)
        choices['training_scenes']=[names[j]for j in train];choices['target']=s['name']
        y=data[s['name']]['y'];unknown=data[s['name']]['unknown']
        choices['methods']={key:dict(action=choices[key],rate=float(q[i,choices[key]]),
            upper=float(np.minimum(1.,y[:,:,choices[key]]+unknown[:,:,choices[key]]).mean()))
            for key in ('ridge','knn','global_fixed','eta13','safety')}
        choices['methods']['crossfit']=crossfit(y)
        choices['methods']['calibration']=calibration(y)
        upper=np.minimum(1.,y+unknown);halves=[np.arange(y.shape[1])[np.arange(y.shape[1])%2==j]for j in (0,1)]
        a0,a1=choices['methods']['crossfit']['actions']
        choices['methods']['crossfit']['upper']=float((upper[:,halves[1],a0].mean()+upper[:,halves[0],a1].mean())/2)
        cal=choices['methods']['calibration'];cal_upper=[]
        for draw,action in zip(cal['draws'],cal['choices']):
            rest=[j for j in range(y.shape[1])if j not in draw['states']]
            cal_upper.append(float(upper[:,rest,action].mean()))
        cal['upper']=float(np.mean(cal_upper))
        a=int(np.argmax(q[i]));choices['methods']['in_sample']=dict(action=a,rate=float(q[i,a]))
        choices['methods']['in_sample']['upper']=float(upper[:,:,a].mean())
        loso[s['name']]=choices
        print('LOSO',s['name'],choices['penalty'],choices['k'],flush=True)
    lofo={}
    for held,train_families in [('bottleneck',('literature','diametric')),('literature',('diametric','bottleneck'))]:
        train=[j for j,s in enumerate(scenes)if s['family']in train_families]
        rows={}
        for i,s in enumerate(scenes):
            if s['family']!=held:continue
            chosen=select(f[train],q[train],f[i],etas)
            chosen['training_scenes']=[names[j]for j in train]
            chosen['methods']={k:dict(action=chosen[k],rate=float(q[i,chosen[k]]))for k in ('ridge','knn','global_fixed','eta13','safety')}
            rows[s['name']]=chosen
        lofo[held]=rows
    extrap=next(s for s in read(HERE/'SCENES.json')if s['family']=='extrapolation')
    prospective=select(f,q,extrap['features'],etas);prospective['training_scenes']=names
    write(HERE/'LOSO_CHOICES.json',loso);write(HERE/'LOFO_CHOICES.json',lofo);write(HERE/'EXTRAPOLATION_FROZEN_CHOICES.json',prospective)
    paths=[HERE/n for n in ('models.py','analyze.py','statistics.py','scenes.py','stages.py','FEATURES.json','LABEL_TABLES.json',
        'LOSO_CHOICES.json','LOFO_CHOICES.json','EXTRAPOLATION_FROZEN_CHOICES.json','SAMPLING_FROZEN.receipt.json',
        'main_CASES.json','main_PLANNED_ROLLOUTS.json','MAIN_PLAN_COVERAGE_AUDIT.json')]
    paths=sorted(set(paths)|set(HERE.glob('*.py'))|{HERE/'PROTOCOL_AMENDMENT.md',HERE/'SCENES.json',
        HERE/'SELECTED_CONTROLLERS.json',HERE/'BACKEND_BUILD.json',HERE/'EXTENSION_BUILD.json',
        HERE/'TASK_PREFIX_CORRECTION.json',HERE/'FEATURE_IMPLEMENTATION_CORRECTION.json',HERE/'IMPORTED_SOURCE_AUDIT.json'})
    paths=sorted(set(paths)|{HERE/n for n in ('PILOT_STALL_RECOVERY.json','SCHEDULER_USER_RECONFIRMATION.json',
        'PILOT_REQUEST_RETAINED.json','WORKER_HEARTBEAT_IMPLEMENTATION.json',
        'CHECKPOINT_BUILD.json','CHECKPOINT_REPLAY_CHECK.json','PILOT_CHECKPOINT_UPGRADE.json',
        'CHECKPOINT_STRICT_BITWISE_AUDIT.json',
        'FIRST_OPERATIONAL_CHECKPOINT_CHECK.json','CHECKPOINT_PLANNER_SCOPE_CHECK.json',
        'CHECKPOINT_STRUCTURAL_READBACK.json','CHECKPOINT_COMPATIBILITY.md',
        'ACTUAL_CHECKPOINT_RESUME_CHECK.json','SHORTER_TIME_LIMIT_CHECK.json',
        'MAIN_PREFLIGHT_UNKNOWN_RETENTION.json','C_CRITERION_CLARIFICATION.json',
        'C_CRITERION_IMPLEMENTATION_CORRECTION.md',
        'EXTRAPOLATION_UNKNOWN_BOUND_ACCOUNTING.json')if(HERE/n).exists()})
    assert all(p.exists()for p in paths)
    if csha is not None:assert sha(cpath)==csha,'C criterion changed during fitting'
    seal('MODEL_FROZEN.receipt.json',paths,before_any_extrapolation_main=True,feature_source='Definition and initial configurations only',
         lambda_grid=[.1,1,10,100],k_grid=[1,2,3,5],tuning='Inner leave-one-training-scene selection utility',
         scene_equal=True,controller_type_equal=True,unknown_lower_failure=True,
         preregistered_C_rule=crule,
         bootstrap='5000 scene -> type -> state paired resamples, frozen prospective policies; local calibration/crossfit selection recomputed',
         ties='Smaller lambda/k; first candidate in canonical order; nearest scene ties use prespecified scene order')


if __name__=='__main__':main()
