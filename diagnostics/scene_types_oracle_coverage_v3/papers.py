"""Versioned native-paper ports on the unchanged common physical tasks."""
from core import *
from runner import case

NAMES={
 'wang2017_consistent_perturbation_di_tracking':'Wang 2017 一致扰动',
 'wang2017_full_di_tracking':'Wang 2017 完整死锁消解',
 'chen_impc_ob_si_tracking':'Chen 2023 动态优先级＋右手规则＋安全走廊',
 'zhang2025_risk_si':'Zhang 2025 自适应风险死锁规避',
 'dac_author_si':'Zhang 2025 死锁感知控制',
 'mgr_author_unicycle_tracking':'Lee 2025 旋转木马'}

def immutable_copy(source,folder,name):
    data=Path(source).read_bytes();checksum=hashlib.sha256(data).hexdigest()
    target=HERE/folder/checksum/name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==data
    else:target.write_bytes(data)
    return target

def make(scene,index,profile,purpose,solo=None):
    assert(HERE/'SCENES.json').exists()
    assert(HERE/'BASELINE_AFTER_FAMILY4.json').exists()
    assert(HERE/'family_4_complete.json').exists(),'Paper rollouts follow the four-column phase'
    method=profile['method'];options=profile['options']
    if '_adapter'in profile:
        adapter=Path(profile['_adapter']);episode=Path(profile['_episode'])
    else:
        prepared=prepare(profile);adapter=Path(prepared['_adapter']);episode=Path(prepared['_episode'])
    c=case(scene,controllers()[0],index,solo=solo,purpose=purpose)
    routes=bool(scene['route_guidance']and method!='chen_impc_ob_si_tracking')
    paths=[adapter,episode]+[HERE/'ports'/n for n in('paper_adapters.py','formula_baselines.py','impc_adapter.py','mgr_geometry.py')]
    c['controller_payload'].update(nominal=dict(type_code=method,parameters=db.canonical(options),checkpoint=''),
        dictionary=None,safety_layer='paper_own_controller_safety',correction_semantics='paper_fixed_profile_common_SI_after_four_column_v3',
        paper_method=method,paper_options=options,paper_route_guidance=routes,paper_guide_variant=False,
        paper_root_diagnostic_only=False,paper_mechanism=True,paper_method_adapter=str(adapter),paper_episode_entry=str(episode),
        paper_execution_sources={str(p):sha(p)for p in paths},paper_source_pins_sha256=sha(HERE/'PAPER_PORT_INPUTS.json'),
        error_policy='failed_agent_zero_one_step_then_normal_call',
        planner_stopping='fixed_termination_condition_checks400_per_original0.04_retry_budget')
    c.update(paper_profile=profile['id'],paper_profile_labels=profile['labels'])
    c['controller_uid']=db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
    c['rollout_uid']=db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
    return c

def version(profile):
    adapter=Path(profile['_adapter'])if '_adapter'in profile else HERE/'paper_methods'/profile['method']/'adapter.py'
    episode=Path(profile['_episode'])if '_episode'in profile else HERE/'paper_episode.py'
    return digest(dict(adapter=sha(adapter),episode=sha(episode),options=profile['options']))[:12]

def prepare(profile):
    profile=dict(profile);method=profile['method']
    profile['_adapter']=str(immutable_copy(HERE/'paper_methods'/method/'adapter.py',f'paper_method_versions/{method}','adapter.py'))
    profile['_episode']=str(immutable_copy(HERE/'paper_episode.py','paper_engine_versions','paper_episode.py'))
    return profile
