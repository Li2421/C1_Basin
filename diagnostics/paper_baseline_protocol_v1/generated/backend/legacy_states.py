"""Paper samplers, physical state inventory, and flow-version geometries."""
from experiment import *
import contextlib
from dataclasses import dataclass, asdict
import io
import math
import numpy as np
from official_sampler import OfficialSampler

@dataclass(frozen=True)
class LiteratureConfig:
    scene:str
    num_agents:int
    agent_radius:float
    dt:float
    max_speed:float
    max_steps:int
    goal_tolerance:float=.08
    wall_radius:float=0.
    wall_collision_margin:float=.005
    agent_collision_margin:float=0.
    def to_dict(self):return asdict(self)

def rect(x0,y0,x1,y1):
    a=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)
    return np.stack([a,np.roll(a,-1,axis=0)],1)

def config(scene,n=None):
    values={'lit_narrow_passage':(3,.15,.2,1.,250),
            'lit_crowded_random':(n,.15,.15,1.,math.ceil(50/.15)),
            'lit_doorway':(2,.1,.1,.3,math.ceil((4*np.sqrt(9+.16)/.3)/.1))}
    if scene=='lit_circle_five':
        receipt=read(HERE/'CIRCLE_SPEED.receipt.json');v=read(HERE/'CIRCLE_SPEED.json')
        assert receipt['files'][str(HERE/'CIRCLE_SPEED.json')]==sha(HERE/'CIRCLE_SPEED.json')
        values[scene]=(5,2.,.05,v['nominal_lqr_speed_p95'],2000)
    return LiteratureConfig(scene,*values[scene])

def walls(scene):
    if scene=='lit_narrow_passage':return np.zeros((0,2,2))
    if scene=='lit_crowded_random':return rect(0,0,2,2)
    if scene=='lit_doorway':return np.concatenate([rect(.9,.3,1.1,1.1),rect(.9,-1.,1.1,-.2)])
    if scene=='lit_circle_five':
        angles=(np.arange(128)+.5)*2*np.pi/128
        vertices=11/np.cos(np.pi/128)*np.stack([np.cos(angles),np.sin(angles)],-1)
        return np.stack([vertices,np.roll(vertices,-1,axis=0)],1)
    raise ValueError(scene)

def paper_states():
    np.random.seed(2026102001)
    result=[]
    def put(scene,index,p,g,n=None):
        p=np.asarray(p,float);g=np.asarray(g,float)
        result.append(dict(scene=scene,index=index,N=len(p),positions=p.tolist(),goals=g.tolist(),velocities=np.zeros_like(p).tolist()))
    put('lit_narrow_passage',0,[[.25,0],[0,.25],[0,-.25]],[[-1,0],[0,.25],[0,-.25]])
    for n in [2,4,6,8,10,12,14]:
        class Scalar:
            def __init__(self,x):self.x=x
            def value(self):return self.x
        sampler=OfficialSampler();sampler.Num=Scalar(n);sampler.r_min=Scalar(.3);sampler.epsilon=Scalar(.1)
        for i in range(100):
            with contextlib.redirect_stdout(io.StringIO()):sampler.set_random_ini();sampler.set_random_target()
            if len(sampler.ini_x)!=n or len(sampler.goal)!=n:raise RuntimeError('Official sampler exhausted its exact retry budget')
            put('lit_crowded_random',i,sampler.ini_x,sampler.goal,n)
    rng=np.random.default_rng(2026102001)
    def circle_set():
        pts=[]
        for _ in range(100000):
            theta=rng.uniform(0,2*np.pi);r=11*np.sqrt(rng.random());p=r*np.array([np.cos(theta),np.sin(theta)])
            if np.linalg.norm(p)+2>11 or any(np.linalg.norm(p-q)<=4 for q in pts):continue
            pts.append(p)
            if len(pts)==5:return pts
        raise RuntimeError('Circle sampler exhausted declared 100000 proposal budget')
    for i in range(100):put('lit_circle_five',i,circle_set(),circle_set())
    put('lit_doorway',0,[[-1,.5],[-1,-.5]],[[2,.1],[2,-.1]])
    return result

def create():
    prereg();states=paper_states();write(HERE/'PAPER_STATES.json',states)
    rng=np.random.default_rng(2026102001)
    flow=list(states)
    for scene,delta in [('lit_narrow_passage',.05),('lit_doorway',.1)]:
        source=next(s for s in states if s['scene']==scene)
        for i in range(1,64):
            for _ in range(100000):
                p=np.asarray(source['positions'])+rng.uniform(-delta,delta,(source['N'],2))
                ii,jj=np.triu_indices(len(p),1)
                if np.all(np.linalg.norm(p[ii]-p[jj],axis=-1)>2*(.15 if scene=='lit_narrow_passage' else .1)):
                    flow.append(dict(source,index=i,positions=p.tolist()));break
            else:raise RuntimeError('Perturbation sampler exhausted')
    write(HERE/'FLOW_STATES_UNCHECKED.json',flow)
    seal('STATES.receipt.json',[HERE/n for n in ('PAPER_STATES.json','FLOW_STATES_UNCHECKED.json','states.py','official_sampler.py')],before_rollouts=True)
    print('paper',len(states),'flow',len(flow))

if __name__=='__main__':create()
