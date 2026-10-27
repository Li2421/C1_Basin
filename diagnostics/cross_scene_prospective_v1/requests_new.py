"""New requests in the frozen make_new and old crowded-request conventions."""
from common import *
from scenes import make_new
from replay import iter_array
import numpy as np

OLD=ROOT/'diagnostics/cross_scene_correction_v1'

@lru_cache(None)
def pool():return read_evidence(V2/'LEARNING_SPLIT_REVISED_20261021_lit_narrow_passage.json')['training_controllers']

@lru_cache(None)
def crowded_templates():
    wanted={c['controller']for c in pool()};out={}
    evidence(OLD/'main_CASES.json')
    for c in iter_array(OLD/'main_CASES.json'):
        ctl=c['controller_payload']['nominal']['controller']
        if c['state']['scene']=='lit_crowded_random' and ctl in wanted and ctl not in out:
            out[ctl]=c
        if len({v['controller_payload']['nominal']['type_code']for v in out.values()})==len(TYPES):break
    # The main table uses one selected instance/type. Second instances have the
    # same profile as that type, with their prespecified nominal parameters.
    assert set(c['type_code']for c in pool())<=set(c['controller_payload']['nominal']['type_code']for c in out.values())
    write(HERE/'CROWDED_N8_REQUEST_TEMPLATES.json',out)
    return out

def request(scene,ctl,state,action,seed=0):
    if scene['family']=='bottleneck':
        base=make_new(scene,ctl,state,action,seed)
        payload={k:v for k,v in base['controller_payload'].items()if k not in ('method','eta_b','theta')}
        payload['execution_sources']=dict(payload['execution_sources'],**{str(HERE/n):sha(HERE/n)for n in
            ('runtime_prospective.py','runtime_prospective_checkpointed.py','prospective_environment.py','controllers_prospective.py')})
        c=make_case(state,base['version'],action['key'],base['scenario_payload'],payload,seed,action['eta_a'],base['state_provenance'])
        c['cross_scene_new']=True
    else:
        templates=crowded_templates();template=templates.get(ctl['controller'])
        if template is None:
            template=next(c for c in templates.values()if c['controller_payload']['nominal']['type_code']==ctl['type_code'])
        scenario=copy.deepcopy(template['scenario_payload']);scenario['config']=dict(scene['config'],num_agents=state['N'])
        payload={k:copy.deepcopy(v)for k,v in template['controller_payload'].items()if k not in ('method','eta_b','theta')}
        payload['nominal']=ctl
        if any(action['eta_a']):
            payload.update(correction_semantics='contact_conflict_relative_gate_20261021',
                correction_revision_receipt_sha256=evidence(V2/'CORRECTION_REVISION.receipt.json'),
                correction_basis='orthoflow3_local_contact_gated',arrived_flag_broadcast=True)
        # With new physical states, retain their own baseline state UID for RNG;
        # the old N8 state's UID must never be broadcast to new initial states.
        c=make_case(state,'flow',action['key'],scenario,payload,seed,action['eta_a'],
            dict(physical_source=scene['physical_source'],scene=scene['name'],index=state['index'],agent=state.get('agent')))
        c['rng_state_uid']=c['state_uid']
    c['prospective_scene']=scene['name'];return c

def capability_cases():
    out=[]
    for scene in read(HERE/'SCENES.json'):
        if not scene['eligible']:continue
        for ctl in pool():
            for state in scene['states']:
                for i in range(state['N']):
                    solo=dict(state,N=1,parent_N=state['N'],agent=i,parent_positions=state['positions'],
                        positions=[state['positions'][i]],goals=[state['goals'][i]],velocities=[state['velocities'][i]],index=state['index']*100+i)
                    for seed in (0,1):out.append(request(scene,ctl,solo,actions()[0],seed))
    return out

def evaluation_cases(stage):
    out=[];chosen=read(HERE/'SELECTED_CONTROLLERS.json');choices=read(HERE/'FROZEN_CHOICES.json');menu={a['key']:a for a in actions()}
    for scene in read(HERE/'SCENES.json'):
        if not scene['eligible']:continue
        keys=['safety','eta13']if stage=='pilot'else sorted(set(choices[scene['name']].values()),key=lambda k:list(menu).index(k))
        for ctl in chosen[scene['name']].values():
            if ctl is None:continue
            for state in scene['states'][:2]if stage=='pilot'else scene['states']:
                for k in keys:out.append(request(scene,ctl,state,menu[k]))
    return out

def prepare(stage):
    for p,h in read(HERE/'MODEL_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
    assert read(HERE/'EXECUTION_CHECK.json')['passed']
    cases=capability_cases()if stage=='capability'else evaluation_cases(stage)
    register(stage,cases)
    # Required public CLI preflight, in addition to the same official planner API.
    import subprocess
    command=[sys.executable,'-B','-m','shared_rollout_db.plan','--manifest',str(HERE/f'{stage}_PLANNED_ROLLOUTS.json'),
        '--output',str(HERE/f'{stage}_OFFICIAL_PREFLIGHT.json')]
    subprocess.run(command,cwd=ROOT,check=True)
    from queueing import build
    build(stage)

if __name__=='__main__':prepare(sys.argv[1])
