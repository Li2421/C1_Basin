from experiment import *
from registry import case,register
from states import config
import numpy as np

def menus():
    p=ROOT/'diagnostics/crosstype6_shared_correction_controller_information_v1/NEW_EVAL_CANDIDATES.json'
    seen={}
    for row in read(p):
        for a in row:
            if a['trigger']=='none' and a['late_source']=='given' and a['switch_mode']=='progress' and a['progress_threshold'] in [.90,.93,.95,.968,.98]:
                key=canonical([a['eta_a'],a['eta_b'],a['progress_threshold']]);seen[key]=dict(eta_a=a['eta_a'],eta_b=a['eta_b'],theta=a['progress_threshold'])
    pool=[seen[k] for k in sorted(seen)];rng=np.random.default_rng(2026102002);indices=rng.choice(len(pool),11,replace=False)
    actions=[dict(eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],theta=None)]+[pool[int(i)] for i in indices]
    write(HERE/'LEARNING_MENU.json',actions)
    seal('LEARNING_MENU.receipt.json',[HERE/'LEARNING_MENU.json'],seed=2026102002,source=str(p),source_sha256=sha(p),pool_size=len(pool),indices=indices.tolist(),before_training_labels=True)
    return actions

def build(stage,scene=None):
    prereg();assert read(HERE/'REPRODUCTION_GATE.json')['passed']
    states=read(HERE/('FLOW_STATES_'+scene+'.json' if scene else 'FLOW_STATES.json'));controllers=read(HERE/'CONTROLLER_MANIFEST.json');rows=[]
    hashes={n:sha(HERE/n) for n in ('flow_runtime.py','flow_scene.py','projection_adapter.py','controllers.py','planning.py','astar_kernel.cpp','cbs_kernel.cpp')}
    eta=read(HERE/'FIXED_ETA.json');actions=[dict(name='safety',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],theta=None)]
    if stage.startswith('fixed'):actions=[dict(name='fixed_eta10',eta_a=eta['10']['values'],eta_b=eta['10']['values'],theta=None),dict(name='fixed_eta13',eta_a=eta['13']['values'],eta_b=eta['13']['values'],theta=None),dict(name='human_progress',eta_a=eta['13']['values'],eta_b=eta['10']['values'],theta=.968)]
    for s in states:
        if stage.startswith('capability'):
            for agent in range(s['N']):
                solo=dict(s,N=1,positions=[s['positions'][agent]],goals=[s['goals'][agent]],velocities=[[0.,0.]],parent_N=s['N'],agent=agent)
                cfg=config(s['scene'],s['N']).to_dict();cfg['num_agents']=1
                if s['scene']=='lit_crowded_random':cfg['max_steps']=int(50/.15)
                sc=dict(version='flow',config=cfg,walls_sha256=sha(HERE/'states.py'),environment_sha256=hashes['flow_scene.py'])
                for ctl in controllers:
                    payload=dict(nominal=ctl,execution_sources=hashes,safety_config='HardProjectionConfig_defaults',precision='jax_x64_false')
                    for seed in (0,1):rows.append(case(solo,'flow','safety',sc,payload,seed))
            continue
        scene=s['scene'];cfg=config(scene,s['N']).to_dict();sc=dict(version='flow',config=cfg,walls_sha256=sha(HERE/'states.py'),environment_sha256=hashes['flow_scene.py'])
        if scene=='lit_crowded_random':cfg['max_steps']=int(50/.15)
        for ctl in controllers:
            for seed in (0,1):
                for a in actions:
                    payload=dict(nominal=ctl,execution_sources=hashes,safety_config='HardProjectionConfig_defaults',precision='jax_x64_false')
                    rows.append(case(s,'flow',a['name'],sc,payload,seed,a['eta_a'],a['eta_b'],a['theta']))
    register(stage,rows)
if __name__=='__main__':build(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
