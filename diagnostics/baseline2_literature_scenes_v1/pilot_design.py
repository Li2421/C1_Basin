from experiment import *
from registry import case,register
from states import config

def manifest(scene):
    result=[]
    for family,name in (('deterministic','确定性行为克隆策略'),('mixture','混合密度行为克隆策略')):
        for seed in (2026103301,2026103302):
            folder=HERE/'pilot_policies'/scene/f'{family}_{seed}';receipt=read(folder/'COMPLETE.receipt.json')
            assert all(sha(p)==h for p,h in receipt['files'].items())
            parameters=dict(folder=str(folder),checkpoint_sha256=sha(folder/'best.msgpack'),normalization_sha256=sha(folder/'NORMALIZATION.json'),config_sha256=sha(folder/'CONFIG.json'))
            result.append(dict(controller=f'{family}_{seed}',type=name,type_code=family,parameters=canonical(parameters),checkpoint=str(folder/'best.msgpack'),seed=seed))
    path=HERE/f'PILOT_CONTROLLER_MANIFEST_{scene}.json'
    if not path.exists():
        write(path,result);seal(f'PILOT_CONTROLLER_MANIFEST_{scene}.receipt.json',[path,HERE/'pilot_design.py'],before_capability_queries=True)
    else:assert read(path)==result
    return result

def make_case(s,ctl,a,seed,slot=None):
    hashes={n:sha(HERE/n) for n in ('flow_runtime.py','flow_scene.py','projection_adapter.py','controllers.py','planning.py','astar_kernel.cpp','cbs_kernel.cpp','pilot_runtime.py','pilot_policies.py')}
    cfg=config(s['scene'],s.get('parent_N',s['N'])).to_dict();cfg['num_agents']=s['N']
    if s['scene']=='lit_crowded_random':cfg['max_steps']=int(50/.15)
    sc=dict(version='flow',config=cfg,walls_sha256=sha(HERE/'states.py'),environment_sha256=hashes['flow_scene.py'])
    payload=dict(nominal=ctl,execution_sources=hashes,safety_config='HardProjectionConfig_defaults',precision='jax_x64_false')
    if slot is not None:payload['capability_agent_slot']=slot
    if a['name']=='nominal_only':payload['execution_sources']=dict(hashes,**{'pilot_nominal_runtime.py':sha(HERE/'pilot_nominal_runtime.py')});payload['safety_config']='not_applied'
    return case(s,'flow',a['name'],sc,payload,seed,a.get('eta_a'),a.get('eta_b'),a.get('theta'))

def main(scene,part):
    prereg();ctls=manifest(scene);states=read(HERE/f'FLOW_STATES_{scene}.json');rows=[];eta=read(HERE/'FIXED_ETA.json')
    safety=dict(name='safety',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],theta=None)
    actions=[safety] if part in ('capability','safety') else [dict(name='nominal_only')] if part=='nominal' else [dict(name='fixed_eta10',eta_a=eta['10']['values'],eta_b=eta['10']['values'],theta=None),dict(name='fixed_eta13',eta_a=eta['13']['values'],eta_b=eta['13']['values'],theta=None),dict(name='human_progress',eta_a=eta['13']['values'],eta_b=eta['10']['values'],theta=.968)]
    for s in states:
        if part=='capability':
            for agent in range(s['N']):
                solo=dict(s,N=1,positions=[s['positions'][agent]],goals=[s['goals'][agent]],velocities=[[0.,0.]],parent_N=s['N'],agent=agent)
                for ctl in ctls:
                    for seed in (0,1):rows.append(make_case(solo,ctl,safety,seed,agent))
        else:
            for ctl in ctls:
                for seed in (0,1):
                    for a in actions:rows.append(make_case(s,ctl,a,seed))
    register(f'pilot_{part}_{scene}',rows)

if __name__=='__main__':main(sys.argv[1],sys.argv[2])
