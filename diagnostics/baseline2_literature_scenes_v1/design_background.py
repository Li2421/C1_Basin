from experiment import *
from registry import register,case
from states import config
def main(scene):
    prereg();states=read(HERE/f'FLOW_STATES_{scene}.json');ctls=read(HERE/'CONTROLLER_MANIFEST.json')
    goal=dict(controller='goal_reference',type='名义朝目标控制器（白盒背景参照）',type_code='goal_reference',parameters='[]')
    eta=read(HERE/'FIXED_ETA.json');fixed=[dict(name='safety',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],theta=None),dict(name='fixed_eta10',eta_a=eta['10']['values'],eta_b=eta['10']['values'],theta=None),dict(name='fixed_eta13',eta_a=eta['13']['values'],eta_b=eta['13']['values'],theta=None),dict(name='human_progress',eta_a=eta['13']['values'],eta_b=eta['10']['values'],theta=.968)]
    nominal=[];whitebox=[];hashes={n:sha(HERE/n) for n in ('flow_runtime.py','flow_scene.py','projection_adapter.py','controllers.py','planning.py','astar_kernel.cpp','cbs_kernel.cpp')}
    for s in states:
        cfg=config(scene,s['N']).to_dict()
        if scene=='lit_crowded_random':cfg['max_steps']=int(50/.15)
        sc=dict(version='flow',config=cfg,walls_sha256=sha(HERE/'states.py'),environment_sha256=hashes['flow_scene.py'])
        for ctl in [*ctls,goal]:
            payload=dict(nominal=ctl,execution_sources=dict(hashes,**{'nominal_runtime.py':sha(HERE/'nominal_runtime.py')}),safety_config='not_applied',precision='jax_x64_false')
            for seed in (0,1):nominal.append(case(s,'flow','nominal_only',sc,payload,seed))
        payload=dict(nominal=goal,execution_sources=hashes,safety_config='HardProjectionConfig_defaults',precision='jax_x64_false')
        for seed in (0,1):
            for a in fixed:whitebox.append(case(s,'flow',a['name'],sc,payload,seed,a['eta_a'],a['eta_b'],a['theta']))
    register('nominal_'+scene,nominal);register('whitebox_'+scene,whitebox)
if __name__=='__main__':main(sys.argv[1])
