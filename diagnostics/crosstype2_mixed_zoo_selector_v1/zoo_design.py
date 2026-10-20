from common import *
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
from registry import register
from generator_reference import normalized
import copy
from functools import lru_cache

DEPENDENCIES=('mixed_execution.py','mixed_runtime.py','mixed_controllers.py','generated_execution.py','rollout_runtime.py','zoo_runtime.py','zoo_models.py','ippo.py','controllers.py','historical_controllers.py','legacy_controllers.py','learned.py','gaussian.py','right_hand.py','trigger.py','progress.py','corrected_astar_reference.py','fast_astar.py','spacetime_astar.py','astar_kernel.so','core.py','common.py','worker.py')
LOCKED=('distributed_cross_entropy','centralized_mppi','implicit')


@lru_cache(None)
def candidate_records():
    return {r['state_uid']:r for r in read(HERE/'CANDIDATES.json')['states']}


@lru_cache(None)
def rhr_rules():
    return read(HERE/'RHR_SHORTLIST.json')


def profiles():
    base=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json')
    source=read(D/'blackbox_families_v1/source_training_protocol.json')
    old=read(D/'blackbox_families_v1/main_protocol.json');config=read(HERE/'NEW_CONTROLLER_CONFIGS.json')
    for name in [f'DM{i}' for i in range(6)]:
        base['profiles'][name]=source['profiles'][name];base['controller_order'].append(name);base['types'][name]='distributed_shooting'
    for family in ('centralized_sampling','cross_entropy'):
        for i in range(6):
            name=f'{family}{i}';base['profiles'][name]=old['profiles'][name];base['controller_order'].append(name);base['types'][name]=family
    for family,kind,grid in [('prioritized_astar','ASTAR_CORRECTED',config['ASTAR']),('distributed_cross_entropy','DCEM',config['CEM']),('centralized_mppi','MPPI',config['MPPI'])]:
        for i,parameters in enumerate(grid):
            name=f'{family}{i}';payload=copy.deepcopy(old['profiles']['cross_entropy0']['payload'])
            payload.update(controller_class=kind,nominal_parameters=parameters,nominal_implementation_sha256=sha(HERE/'mixed_controllers.py'),execution_semantics=f'crosstype2_{family}_TT_v1',chain='TT',placement='TT',controller_config_sha256=sha(HERE/'NEW_CONTROLLER_CONFIGS.json'))
            if kind=='ASTAR_CORRECTED':payload['astar_sources']={n:sha(HERE/n) for n in ('corrected_astar_reference.py','fast_astar.py','spacetime_astar.py','astar_kernel.so')}
            base['profiles'][name]=dict(controller_uid=uid('ctl',payload),payload=payload);base['controller_order'].append(name);base['types'][name]=family
    for cfg in read(HERE/'IPPO_TRAINING_CONFIGS.json'):
        folder=HERE/'policies'/cfg['name'];receipt=folder/'CHECKPOINT_SELECTED.receipt.json'
        if not receipt.exists():continue
        for path,digest in read(receipt)['files'].items():assert sha(path)==digest
        payload=copy.deepcopy(base['profiles'][cfg['source_instance']]['payload'])
        payload.update(controller_class='ring_ippo',flow_checkpoint_path=str(folder/'best.msgpack'),flow_checkpoint_sha256=sha(folder/'best.msgpack'),normalization_sha256=sha(folder/'NORMALIZATION.json'),policy_config_sha256=sha(folder/'CONFIG.json'),observation_frame=cfg['frame'],adapter_manifest_path=cfg['manifest'],placement='TT',chain='TT',jax_enable_x64=False,conditioning='nominal_TT_project_correct_reproject',execution_semantics='crosstype2_conservative_ippo_TT_v1',nominal_implementation_sha256=sha(HERE/'zoo_runtime.py'),model_implementation_sha256=sha(HERE/'ippo.py'))
        base['profiles'][cfg['name']]=dict(controller_uid=uid('ctl',payload),payload=payload);base['controller_order'].append(cfg['name']);base['types'][cfg['name']]='ring_ippo_conservative'
    return base


def candidate_action(base,index,state_uid=None):
    if index<10:
        record=candidate_records()[state_uid]
        return record['actions'][index]
    zero=[0.,0.,0.];eta=base['eta']
    if index==10:return dict(name='safety_baseline',eta_a=zero,eta_b=zero,switch_mode='fixed')
    if index in (11,12):return dict(name='S500' if index==11 else 'Sprog',eta_a=eta[13],eta_b=eta[10],switch_mode='step500' if index==11 else 'progress')
    if index in (13,14,15):
        key=('fixed','RHR_b','RHR_c')[index-13];rule=rhr_rules()[key]
        return dict(name=('RHR','RHR_b','RHR_c')[index-13],eta_a=zero,eta_b=zero,switch_mode='fixed',rhr_rule=rule)
    if index==16:return dict(name='nominal_only',eta_a=zero,eta_b=zero,switch_mode='fixed')
    if index==17:return dict(name='F10',eta_a=eta[10],eta_b=eta[10],switch_mode='fixed')
    if index==18:return dict(name='F13',eta_a=eta[13],eta_b=eta[13],switch_mode='fixed')
    raise ValueError(index)


def case(base,name,s,index,seed=0,probe=False):
    a=candidate_action(base,index,s['state_uid']);v=copy.deepcopy(base['profiles'][name]['payload'])
    if index==16:v.update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='round38_nominal_only_v1')
    elif index<10 and a['switch_mode']=='progress':
        v['generator_progress_schedule']=dict(early_eta='request_eta_uid',late_eta=a['eta_b'],pstar=.9681620884830543,fallback=600,progress_implementation_sha256=sha(HERE/'progress.py'),wrapper_sha256=sha(HERE/'generated_execution.py'))
    elif index==12:v['progress_schedule']=dict(pstar=.9681620884830543,fallback=600,implementation_sha256=sha(HERE/'progress.py'))
    elif index==11:v['eta_schedule']=dict(eta_a=a['eta_a'],eta_b=a['eta_b'],switch_step=500,action_index_rule='a if index<t else b')
    elif a.get('rhr_rule'):v['continuing_right_hand']=dict(rule=a['rhr_rule'],implementation_sha256=sha(HERE/'right_hand.py'),trigger_sha256=sha(HERE/'trigger.py'))
    if probe:v.update(success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_steps=300,terminate_on_success=False,terminate_on_collision=False,probe_code_sha256=sha(HERE/'rollout_runtime.py'))
    cid=uid('ctl',v);eid=eta_identity(a['eta_a'])[0];sk=canonical(dict(future_index=seed))
    c=dict(rollout_uid=uid('probe' if probe else 'roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk)),state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=name,family=base['types'][name],arm=a['name'],candidate_index=index,seed=seed,eta_a=a['eta_a'],eta_b=a['eta_b'],switch_mode=a['switch_mode'],switch_step=500 if a['switch_mode']=='step500' else 700,state_index=s.get('study_index',0),placement='FF' if base['types'][name]=='Flow' else 'TT',rhr_rule=a.get('rhr_rule'))
    return c,dict(controller_uid=cid,payload=v)


def freeze(stage,p,rows,cps,**kw):
    register(stage,p,rows,cps)
    seal(f'{stage}_EXECUTION_FROZEN.receipt.json',[HERE/f'{stage}_protocol.json',HERE/f'{stage}_CASES.json',HERE/f'{stage}_PREFLIGHT.receipt.json',*[HERE/n for n in DEPENDENCIES]],**kw)


def gate(cohort):
    prereg();assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed']
    assert read(HERE/'MIXED_CONTROLLER_TESTS.json')['passed'];assert read(HERE/'CORRECTED_ASTAR_TESTS.json')['passed']
    base=profiles();names=[n for n in base['controller_order'] if (base['types'][n] in LOCKED)==(cohort=='locked') and base['types'][n] in ('distributed_shooting','centralized_sampling','cross_entropy','prioritized_astar','ring_ippo_conservative',*LOCKED)]
    # PPO is added in a separate gate after its original checkpoint rule is done.
    if cohort=='ppo':names=[n for n in base['controller_order'] if base['types'][n]=='ring_ippo_conservative']
    elif cohort=='development':names=[n for n in names if base['types'][n]!='ring_ippo_conservative']
    assert names
    states=read(D/'blackbox_families_v1/PILOT_STATES.json');p={**base,'states':states,'controller_order':names,'phase':'capability'};rows=[];cps={}
    for name in names:
        for s in states:
            for index in (10,12,17,13):
                c,cp=case(base,name,s,index);rows.append(c);cps[cp['controller_uid']]=cp
    freeze('gate_'+cohort,p,rows,cps,locked_rates_blinded=cohort=='locked',split='capability')


def gate_collect(cohort):
    stage='gate_'+cohort;assert (HERE/f'{stage}_COMPLETE.receipt.json').exists()
    p=read(HERE/f'{stage}_protocol.json');cases=read(HERE/f'{stage}_CASES.json');values={}
    with rollout_db.connect(True) as db:
        for c in cases:
            r=db.execute('SELECT success,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None
            v=values.setdefault(c['base_controller_name'],{}).setdefault(c['arm'],[0,0,0]);v[0]+=int(r['success'] and not r['numerical_failure']);v[1]+=int(r['numerical_failure']);v[2]+=1
    rows=[]
    for name,arms in values.items():
        assert all(v[2]==16 for v in arms.values());passed=max(v[0]/v[2] for v in arms.values())>=.15
        row=dict(controller=name,type=p['types'][name],passed=passed,excluded=not passed,reason=None if passed else 'Maximum four-arm lower success rate below 15%; no replacement')
        if cohort!='locked':row.update(arms={a:dict(success=v[0],unknown=v[1],n=v[2]) for a,v in arms.items()},no_repair_needed=arms['safety_baseline'][0]/16>=.98)
        rows.append(row)
    write(HERE/f'{cohort.upper()}_CAPABILITY.json',dict(utc=utc(),locked_rates_blinded=cohort=='locked',rows=rows,eligible=[r['controller'] for r in rows if r['passed']]))
    print(cohort,'capability statuses saved;',sum(r['passed'] for r in rows),'eligible;',len(rows),'total; locked arm rates not emitted' if cohort=='locked' else '')


def training(cohort):
    prereg();base=profiles();old=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json')
    if cohort=='legacy':names=old['controller_order']
    else:
        capability=read(HERE/('PPO_CAPABILITY.json' if cohort=='ppo' else 'DEVELOPMENT_CAPABILITY.json'))
        names=capability['eligible']
    assert cohort in ('legacy','mixed','ppo')
    states=read(HERE/'TRAINING_STATES.json');p={**base,'states':states,'controller_order':names,'phase':'training'};rows=[];cps={}
    for name in names:
        for si,s in enumerate(states):
            for index in range(16):
                c,cp=case(base,name,s,index);c['state_index']=si;c['phase_state_index']=si
                rows.append(c);cps[cp['controller_uid']]=cp
    # Demonstrate exact identity reuse for every old learned seed0 observation.
    if cohort=='legacy':
        with np.load(D/'crosstype1b_generator_critic_pipeline_v1/TRAINING_BUNDLE.npz',allow_pickle=False) as z:
            indices={str(n):i for i,n in enumerate(z['names'])}
            historical_uids=z['uids'].copy()
            for c in rows:
                if c['candidate_index']>=13:continue
                assert c['rollout_uid']==str(historical_uids[indices[c['base_controller_name']],c['state_index'],c['candidate_index'],0]),('legacy UID mismatch',c)
    stage='training_'+cohort
    freeze(stage,p,rows,cps,split='training',source_gate='historical62' if cohort=='legacy' else str(HERE/('PPO_CAPABILITY.json' if cohort=='ppo' else 'DEVELOPMENT_CAPABILITY.json')))


def training_budget():
    phases=['training_legacy','training_mixed','training_ppo'];values={}
    for phase in phases:
        path=HERE/f'{phase}_MISSING_CASES.json'
        if not path.exists():continue
        rows=read(path);values[phase]=len(rows)
    assert len(values)==3,'Exact full training ceiling requires all three preflights'
    write(HERE/'TRAINING_FINAL_PREFLIGHT_BUDGET.json',dict(utc=utc(),planned_missing=sum(values.values()),phases=values,cap=sum(values.values()),initial_estimate=35000,development_candidate_additional_cap=8000,unknown_reruns=0))
    seal('TRAINING_FINAL_PREFLIGHT_BUDGET.receipt.json',[HERE/'TRAINING_FINAL_PREFLIGHT_BUDGET.json',*[HERE/f'{p}_PREFLIGHT.receipt.json' for p in phases]],before_training_label_submission=True)


if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
