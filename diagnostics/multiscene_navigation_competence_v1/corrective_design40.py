"""Exact39-dimensional nominal identities and original-state multiagent capability gates."""
from common import *
from design_generated import identities as base_identities,ARMS
from collect39 import policy_assets,checkpoint
from registry import register
from shared_rollout_db.src.rollout_db import uid,canonical,connect

def require_ready(family):
    phase=read(HERE/'FULL_NAVIGATION_PHASE_COMPLETE.json');assert phase['correction_stage_eligible']
    pointer=read(HERE/'CORRECTIVE_RUNTIME39_CURRENT.json');receipt=HERE/pointer['receipt'];assert sha(receipt)==pointer['sha256']
    assert all(sha(path)==expected for path,expected in read(receipt)['files'].items())
    test=read(HERE/'NOMINAL39_CALLBACK_UNIT_TEST.json');assert test['passed']
    assert test.get('callback_sha256')==sha(HERE/'nominal39_callback.py') and test.get('generated_runtime_sha256')==sha(HERE/'generated_runtime.py')
    # Random-obstacle navigation training is allowed; every corrective label stays locked.
    if family=='G':
        assert (HERE/'AF_ANALYSIS_FROZEN.receipt.json').exists() and (HERE/'FINAL_MODEL_FROZEN.receipt.json').exists()

def identities(state,arm,seed,cp,purpose='task'):
    assert len(state['positions'])==4 and purpose in ['task','probe']
    complete=read(Path(cp).parent/'COMPLETE.receipt.json')
    for p,h in complete['files'].items():assert sha(p)==h
    c,rows=base_identities(state,'learned',arm,seed,cp,state['H'])
    row=next(v for table,v in rows if table=='controller_config');profile=json.loads(row['config_json'])
    profile.update(observation_dimensions=39,nominal_source='39-dimensional fixed field-waypoint policy',nominal_sources={n:sha(HERE/n) for n in ['nominal39_callback.py','policies39.py','observation39.py','navigation_field.py','batch_fields39.py','multiagent_worker40.py']},chain_sources={n:sha(HERE/n) for n in ['generated_runtime.py','controllers.py','trigger.py']},**policy_assets(cp))
    if purpose=='probe':profile.update(probe_steps=200,termination_flags_off=True,contains_task_labels=False,probe_sources={'probe_measurements40.py':sha(HERE/'probe_measurements40.py')})
    cid=uid('ctl',profile);row.update(controller_uid=cid,config_json=canonical(profile),conditioning_version='39dim_nominal_goalstop_projection_correction_projection')
    if purpose=='probe':row['success_semantics_version']='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES'
    c.update(controller_uid=cid,observation_dimensions=39)
    c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=cid,seed=c['seed_key']))
    if purpose=='probe':c.update(probe_steps=200,contains_task_labels=False)
    return c,rows

def qualified_navigation(family):
    gate=read(HERE/f'NAVIGATION_CAPABILITY_{family}.json')
    return [r for r in gate['instances'] if r['passed']]

def prepare_capability(family):
    require_ready(family);assert read(HERE/f'NAVIGATION_CAPABILITY_{family}.json')['family_passed']
    states=[s for s in read(HERE/'ROUND39_CATALOG_FROZEN.json')['catalog'][family] if s['role']=='pilot'];assert len(states)==16
    cases=[];rows=[]
    for instance in qualified_navigation(family):
        cp=checkpoint(family,instance['kind'],instance['seed'],3);name=f"{family}_{instance['kind']}_{instance['seed']}"
        for s in states:
            for arm in ['safety_baseline','F10','S13']:
                c,r=identities(s,arm,0,cp);c['controller_name']=name;cases.append(c);rows+=r
    stage='multiagent40_'+family;register(stage,cases,rows)
    seal(f'MULTIAGENT40_{family}_FROZEN.receipt.json',[HERE/'ROUND39_CATALOG_FROZEN.json',HERE/'corrective_design40.py',HERE/'multiagent_worker40.py',HERE/'generated_runtime.py',HERE/'nominal39_callback.py',HERE/'NOMINAL39_CALLBACK_UNIT_TEST.json'],before_family_capability_labels=True,maximum_success_threshold=.2,trivial_all_arms_threshold=.98,no_replacement=True)
    return stage

def analyze_capability(family):
    stage='multiagent40_'+family;assert (HERE/f'{stage}_POSTFLIGHT.json').exists();cases=read(HERE/f'{stage}_CASES.json');results=[]
    with connect(True) as db:
        for name in sorted({c['controller_name'] for c in cases}):
            stats={}
            for arm in ['safety_baseline','F10','S13']:
                selected=[c for c in cases if c['controller_name']==name and c['arm']==arm];assert len(selected)==16
                raw=[dict(db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()) for c in selected]
                counts={k:sum(r['numerical_failure'] if k=='unknown' else r[k] for r in raw) for k in ['success','collision','timeout','unknown']}
                stats[arm]=dict(episodes=16,**counts,lower=counts['success']/16,upper=(counts['success']+counts['unknown'])/16)
            passed=max(x['lower'] for x in stats.values())>=.2 and not all(x['lower']>.98 for x in stats.values())
            reason='eligible' if passed else 'all three arms exceed98% observed success' if all(x['lower']>.98 for x in stats.values()) else 'maximum observed success below20%; no instance replacement'
            upper_passed=max(x['upper'] for x in stats.values())>=.2 and not all(x['upper']>.98 for x in stats.values())
            results.append(dict(name=name,arms=stats,passed=passed,upper_passed=upper_passed,gate_uncertain=passed!=upper_passed,reason=reason))
    eligible=[r['name'] for r in results if r['passed']];p=HERE/f'MULTIAGENT_CAPABILITY_{family}.json'
    write(p,dict(family=family,instances=results,eligible=eligible,family_passed=len(eligible)>=2))
    seal(f'MULTIAGENT_CAPABILITY_{family}.receipt.json',[p,HERE/f'{stage}_POSTFLIGHT.json'],no_replacement=True,no_task_outcome_retries=True)
    return eligible
