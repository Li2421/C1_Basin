"""No-rollout checks of shared caches and numerical reporting contracts."""
from experiment import *
import ast
import multiprocessing as mp
import numpy as np

def writer(path,owner):
    from atomic_write import write
    for iteration in range(40):write(path,dict(owner=owner,iteration=iteration,payload=[owner]*256))

def main():
    from derived_metrics import weighted_rates,controller_drops
    from statistics import contrast
    path=HERE/'tmp'/'ATOMIC_CONCURRENCY_TEST.json'
    ctx=mp.get_context('spawn');processes=[ctx.Process(target=writer,args=(path,i)) for i in range(4)]
    for p in processes:p.start()
    for p in processes:p.join();assert p.exitcode==0
    final=read(path);assert final['iteration']==39 and final['payload']==[final['owner']]*256
    assert not list(path.parent.glob(path.name+'.tmp.*'))
    rows=[]
    # Type A has two controllers, B one; each physical state has both seeds.
    for t,ctl,states,value in [('A','a',1,1),('A','b',3,0),('B','c',2,1)]:
        for state in range(states):
            for seed in (0,1):
                rows.append(dict(layer='distributed',method='safety',type=t,controller=ctl,scene='synthetic',N=2,state=state,seed=seed,
                    success=value,upper=value,collision=0,timeout=1-value,unknown=0,deadlock_episode=0,wall_episode=0,canonical_joint_episode=0,actual_joint_episode=0,deadlock_agents=0,wall_agents=0))
    rate=weighted_rates(rows)['distributed|safety'];assert rate['rates']['success']==.75 and rate['controllers']==3
    # Unknown bounds must not be confused with ordinary lower-bound contrasts.
    arms=[('distributed','safety'),('distributed','fixed_eta10')]
    point=np.asarray([.8,.9,.6,.85]);result=contrast(point,np.tile(point,(20,1)),arms,{arms[1]:1,arms[0]:-1})
    assert np.allclose(result['identified_bounds'],[-.3,.05]) and np.isclose(result['delta'],-.2)
    target=[dict(r,method='fixed_eta10',success=0,upper=1) for r in rows if r['controller']=='a']
    drop=next(r for r in controller_drops(rows+target) if r['controller']=='a' and r['method']=='fixed_eta10' and r['metric']=='harm_vs_safety')
    assert not drop['definite_drop_10pp'] and drop['possible_drop_10pp']
    import learn_critic,freeze_scene_choices
    assert learn_critic.SEEDS==(17,23,41)
    syntax=[]
    for source in HERE.glob('*.py'):ast.parse(source.read_text());syntax.append(source.name)
    write(HERE/'DELIVERY_COMPONENT_TESTS.json',dict(utc=utc(),passed=True,atomic_concurrent_processes=4,atomic_writes=160,type_equal_unbalanced_example=.75,unknown_contrast_bounds=result['identified_bounds'],critic_and_choice_imports=True,python_sources_parsed=syntax,new_rollouts=0,new_training=0))
    print('Delivery component checks passed',flush=True)

if __name__=='__main__':main()
