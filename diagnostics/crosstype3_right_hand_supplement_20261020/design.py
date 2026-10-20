"""Exact historical profile identities and preregistered new nominal recipes."""
from common import *
from registry import register
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy

def base_unseen():
    base=read(HERE/'BASE_PROTOCOL.json');base=copy.deepcopy(base)
    assert read(HERE/'USER_DESIGN_DECISIONS.json')['user_answer']=='按此前置例外执行'
    for cfg in read(HERE/'TRANSFORMER_TRAINING_CONFIGS.json'):
        folder=HERE/'policies'/cfg['name'];r=read(folder/'COMPLETE.receipt.json')
        for path,h in r['files'].items():assert sha(path)==h,path
        v=copy.deepcopy(base['profiles'][cfg['source_recipe']]['payload'])
        v.update(controller_class='transformer_bc',flow_checkpoint_path=str(folder/'best.msgpack'),flow_checkpoint_sha256=sha(folder/'best.msgpack'),normalization_sha256=sha(folder/'NORMALIZATION.json'),policy_config_sha256=sha(folder/'CONFIG.json'),observation_frame=cfg['frame'],adapter_manifest_path=cfg['manifest'],placement='TT',chain='TT',jax_enable_x64=False,conditioning='nominal_TT_project_correct_reproject',execution_semantics='crosstype3_transformer_bc_TT_v1',nominal_implementation_sha256=sha(HERE/'new_execution.py'),model_implementation_sha256=sha(HERE/'unseen_models.py'),transformer_layers=cfg['layers'])
        base['profiles'][cfg['name']]=dict(controller_uid=uid('ctl',v),payload=v);base['types'][cfg['name']]='transformer_bc'
    for kind,file in [('gradient_mpc','gradient_mpc.py'),('conflict_based_search','conflict_based_search.py')]:
        for i,params in enumerate(read(HERE/'UNSEEN_CONTROLLER_CONFIGS.json')[kind]):
            name=f'{kind}{i}';v=copy.deepcopy(base['profiles']['DM0']['payload'])
            v.update(controller_class=kind,nominal_parameters=params,nominal_implementation_sha256=sha(HERE/file),execution_semantics=f'crosstype3_{kind}_TT_v1',controller_config_sha256=sha(HERE/'UNSEEN_CONTROLLER_CONFIGS.json'))
            if kind=='conflict_based_search':v['cbs_sources']={n:sha(HERE/n) for n in ('cbs_kernel.so','cbs_kernel.cpp','conflict_based_search.py','spacetime_astar.py','fast_astar.py')}
            base['profiles'][name]=dict(controller_uid=uid('ctl',v),payload=v);base['types'][name]=kind
    base['controller_order']=read(HERE/'OPENING_CONFIG.json')['unseen_order'];return base

def anchor(base,name):
    zero=[0.,0.,0.];eta=base['eta']
    if name in ('safety_baseline','nominal_only'):return dict(name=name,eta_a=zero,eta_b=zero,switch_mode='fixed',switch_step=700)
    if name in ('F10','F13'):
        e=eta[int(name[1:])];return dict(name=name,eta_a=e,eta_b=e,switch_mode='fixed',switch_step=700)
    assert name in ('S500','Sprog')
    return dict(name=name,eta_a=eta[13],eta_b=eta[10],switch_mode='step500' if name=='S500' else 'progress',switch_step=500 if name=='S500' else 700)

def action_case(base,name,s,a,seed,state_index,candidate_index=-1,probe=False):
    v=copy.deepcopy(base['profiles'][name]['payload']);mode=a['switch_mode']
    if a['name']=='nominal_only':v.update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='round38_nominal_only_v1')
    elif mode=='progress':
        v['generator_progress_schedule']=dict(early_eta='request_eta_uid',late_eta=a['eta_b'],pstar=.9681620884830543,fallback=600,progress_implementation_sha256=sha(HERE/'progress.py'),wrapper_sha256=sha(HERE/'generated_execution.py'))
    elif mode in ('step','step500'):
        v['eta_schedule']=dict(eta_a=a['eta_a'],eta_b=a['eta_b'],switch_step=a['switch_step'],action_index_rule='a if index<t else b')
    if probe:v.update(success_semantics='UNLABELED_BEHAVIOR_NO_TASK_OUTCOMES',probe_steps=300,terminate_on_success=False,terminate_on_collision=False,probe_code_sha256=sha(HERE/'rollout_runtime.py'))
    cid=uid('ctl',v);eid=eta_identity(a['eta_a'])[0];sk=canonical(dict(future_index=seed));rid=uid('probe' if probe else 'roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk))
    c=dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=name,family=base['types'][name],arm=a['name'],candidate_index=int(candidate_index),seed=seed,eta_a=a['eta_a'],eta_b=a['eta_b'],switch_mode=mode,switch_step=a['switch_step'] if mode in ('step','step500') else 700,state_index=state_index,placement='FF' if base['types'][name]=='Flow' else 'TT',rhr_rule=None)
    return c,dict(controller_uid=cid,payload=v)

def freeze(stage,p,rows,profiles):
    prereg()
    for name in ('EXECUTION_REPRODUCTION','GENERATOR_REPRODUCTION','TRANSFORMER_UNIT_TESTS','OPTIMIZER_UNIT_TESTS'):assert read(HERE/f'{name}.json')['passed']
    assert (HERE/'HISTORY_AUDIT.receipt.json').exists()
    register(stage,p,rows,profiles)
    missing=read(HERE/f'{stage}_MISSING_CASES.json')
    write(HERE/f'{stage}_EXACT_BUDGET.json',dict(utc=utc(),planned=len(rows),exact_missing_cap=len(missing),only_genuine_missing=True,unknown_reruns=0))
    paths=[HERE/f'{stage}_{n}' for n in ('protocol.json','CASES.json','PREFLIGHT.receipt.json','EXACT_BUDGET.json')]
    deps=('design.py','worker.py','new_execution.py','unseen_models.py','gradient_mpc.py','conflict_based_search.py','mixed_execution.py','mixed_runtime.py','mixed_controllers.py','generated_execution.py','rollout_runtime.py','zoo_runtime.py','zoo_models.py','ippo.py','controllers.py','historical_controllers.py','legacy_controllers.py','learned.py','gaussian.py','right_hand.py','trigger.py','progress.py','corrected_astar_reference.py','fast_astar.py','spacetime_astar.py','astar_kernel.so','cbs_kernel.so','core.py','common.py','registry.py','probe_prefix.py','history_audit.py','resource_guard.py')
    seal(f'{stage}_EXECUTION_FROZEN.receipt.json',paths+[HERE/n for n in deps],before_submission=True,locked_correction_rates_blinded=p['phase'] in ('capability','probe'))

def training():
    base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');candidates=read(HERE/'CANDIDATES.json')['states'][:48]
    with np.load(HERE/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:indices=z['candidate_indices'];order=list(z['names'])
    rows=[];profiles={};old=D/'crosstype1c_two_eta_schedule_v1'
    with np.load(old/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:old_names=list(z['names']);old_uids=z['uids']
    for name in base['controller_order']:
        i=order.index(name)
        for si,s in enumerate(states):
            for slot,ci in enumerate(indices[i,si]):
                c,cp=action_case(base,name,s,candidates[si]['actions'][int(ci)],0,si,int(ci));c.update(training_slot=slot,phase_state_index=si)
                if name in old_names:assert c['rollout_uid']==str(old_uids[old_names.index(name),si,slot,0]),('Historical identity mismatch',name,si,slot)
                rows.append(c);profiles[cp['controller_uid']]=cp
    freeze('training',dict(base,states=states,phase='training'),rows,profiles)
    assert len(read(HERE/'training_MISSING_CASES.json'))<=52416

def capability():
    base=base_unseen();states=read(D/'blackbox_families_v1/PILOT_STATES.json');rows=[];profiles={}
    for name in base['controller_order']:
        for si,s in enumerate(states):
            for arm in ('safety_baseline','Sprog','F13'):
                c,cp=action_case(base,name,s,anchor(base,arm),0,si);rows.append(c);profiles[cp['controller_uid']]=cp
    freeze('capability',dict(base,states=states,phase='capability'),rows,profiles)

def capability_collect():
    assert (HERE/'capability_COMPLETE.receipt.json').exists()
    p=read(HERE/'capability_protocol.json');counts={}
    with rollout_db.connect(True) as db:
        for c in read(HERE/'capability_CASES.json'):
            r=db.execute('SELECT success,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None
            v=counts.setdefault(c['base_controller_name'],{}).setdefault(c['arm'],[0,0]);v[0]+=int(r['success'] and not r['numerical_failure']);v[1]+=1
    rows=[]
    for name,arms in counts.items():
        assert set(arms)=={'safety_baseline','Sprog','F13'} and all(v[1]==16 for v in arms.values())
        passed=max(v[0]/16 for v in arms.values())>=.15;safety=arms['safety_baseline'][0]/16
        rows.append(dict(controller=name,type=p['types'][name],passed=passed,safety_success_rate=safety,no_repair_needed=safety>=.98))
    write(HERE/'UNSEEN_CAPABILITY.json',dict(utc=utc(),rows=rows,eligible=[r['controller'] for r in rows if r['passed']],correction_rates_not_emitted=True))
    seal('UNSEEN_CAPABILITY.receipt.json',[HERE/'UNSEEN_CAPABILITY.json',HERE/'capability_COMPLETE.receipt.json',HERE/'design.py'],correction_rates_blinded=True)
    print('Blinded capability statuses sealed; individual and aggregate correction success rates not emitted',flush=True)

def probes():
    base=base_unseen();eligible=read(HERE/'UNSEEN_CAPABILITY.json')['eligible'];base['controller_order']=eligible
    states=read(D/'crosstype1_learned_zoo_selector_v1/locked_probe_protocol.json')['states'];rows=[];profiles={}
    for name in eligible:
        for si,s in enumerate(states):
            for arm in ('safety_baseline','F13','F10'):
                c,cp=action_case(base,name,s,anchor(base,arm),0,si,probe=True);rows.append(c);profiles[cp['controller_uid']]=cp
    freeze('probe',dict(base,states=states,phase='probe'),rows,profiles)

if __name__=='__main__':globals()[sys.argv[1]]()
