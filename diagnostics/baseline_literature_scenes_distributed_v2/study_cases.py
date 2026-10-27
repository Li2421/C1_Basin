"""Exact v1 physical identity where unchanged; separate v2 execution identities."""
from experiment import *
from states import config
from registry import case,register
from functools import lru_cache

@lru_cache(None)
def profiles():
    from source_inputs import index
    old_names=('flow_runtime.py','flow_scene.py','projection_adapter.py','controllers.py','planning.py','astar_kernel.cpp','cbs_kernel.cpp')
    old={n:index()[str(V1/n)] for n in old_names}
    frozen=read(HERE/'IMPLEMENTATION_FROZEN.receipt.json')
    return old,{Path(p).name:h for p,h in frozen['files'].items()},index()[str(V1/'states.py')]

def make(s,ctl,layer='distributed',method='safety',seed=0,eta_a=None,eta_b=None,theta=None):
    old,new,sampler=profiles();scene=s['scene'];cfg=config(scene,s.get('parent_N',s['N'])).to_dict();cfg['num_agents']=s['N']
    changed=scene=='lit_crowded_random'
    scenario=dict(version='flow',config=cfg,walls_sha256=sha(HERE/'states.py') if changed else sampler,environment_sha256=sha(HERE/'flow_scene.py') if changed else old['flow_scene.py'])
    provenance=dict(sampler=sampler,state_index=s['index'])
    if changed:provenance.update(geometry_revision=sha(HERE/'flow_scene.py'),source='v1 frozen physical samples; no walls',parent_N=s.get('parent_N',s['N']),agent=s.get('agent'))
    if layer=='central' and not changed:
        payload=dict(nominal=ctl,execution_sources=old,safety_config='HardProjectionConfig_defaults',precision='jax_x64_false')
    else:
        payload=dict(nominal=ctl,execution_sources=new,safety_config='HardProjectionConfig_defaults',precision='jax_x64_false',safety_layer=layer,correction_basis='orthoflow3_local' if layer=='distributed' else 'orthoflow3_joint')
    c=case(s,'flow',method,scenario,payload,seed,eta_a,eta_b,theta,provenance)
    c['safety_layer']=layer
    return c

def qualified(scene):
    names={'lit_narrow_passage':'capability_v2_lit_narrow_passage','lit_crowded_random':'capability_lit_crowded_random','lit_doorway':'capability_lit_doorway','lit_circle_five':'capability_lit_circle_five'}
    return {r['controller'] for r in read(HERE/(names[scene]+'_QUALIFICATION.json')) if r['qualified']}

def build_capability():
    scene='lit_crowded_random';rows=[]
    for ctl in read(HERE/'CONTROLLER_MANIFEST.json'):
        for s in read(HERE/'CAPABILITY_STATES_lit_crowded_random.json'):
            for agent in range(s['N']):
                solo=dict(s,N=1,positions=[s['positions'][agent]],goals=[s['goals'][agent]],velocities=[[0.,0.]],parent_N=s['N'],agent=agent)
                for seed in (0,1):rows.append(make(solo,ctl,seed=seed))
    assert len(rows)==read(HERE/'STUDY_DESIGN.json')['capability_crowded_requests']
    register('capability_'+scene,rows)

def build_stage(layer,kind,scene):
    rows=[];ctls=read(HERE/'CONTROLLER_MANIFEST.json');fixed=read(HERE/'FIXED_ETA.json')
    if kind=='fixed':ctls=[c for c in ctls if c['controller'] in qualified(scene)]
    actions=[dict(method='safety',eta_a=[0.,0.,0.])]
    if kind=='fixed':actions=[dict(method='fixed_eta10',eta_a=fixed['10']['values']),dict(method='fixed_eta13',eta_a=fixed['13']['values']),dict(method='human_progress',eta_a=fixed['13']['values'],eta_b=fixed['10']['values'],theta=.968)]
    if kind=='whitebox':ctls=[dict(controller='goal_reference',type='名义朝目标控制器',type_code='goal_reference',parameters='[]',checkpoint='',family='whitebox')]
    if kind=='nominal':actions=[dict(method='nominal_only',eta_a=[0.,0.,0.])]
    for ctl in ctls:
        for s in read(HERE/f'FLOW_STATES_{scene}.json'):
            for a in actions:
                for seed in (0,1):rows.append(make(s,ctl,layer,seed=seed,**a))
    register(f'{layer}_{kind}_{scene}',rows)

if __name__=='__main__':
    if sys.argv[1]=='capability':build_capability()
    else:build_stage(*sys.argv[1:])
