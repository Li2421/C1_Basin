"""Separate native callback clocks from the unchanged scenario score clock.

Exact SI ZOH segments are evaluated analytically. No rendering interpolation,
no CBF retuning, no repeated Euler substeps used as a claimed acceleration.
"""
from native_io import *
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
OLD=ROOT/'diagnostics/c1_v3_new_baseline_reproduction_20261103'
sys.path.insert(0,str(OLD))
from common_plant import CommonPlant,point_segment_distance,segment_distance
PERIODS={'dac':.1,'mgr':.05,'mcbf':.05,'circulation':.02,'chen':.15}

def frac(x):return Fraction(str(x))

def timeline(scenario,period):
    score=frac(scenario['config']['dt']);horizon=score*scenario['config']['max_steps'];p=frac(period)
    controls={i*p:i for i in range(int((horizon- Fraction(1,10**12))//p)+1)}
    scores={i*score for i in range(1,scenario['config']['max_steps']+1)}
    return sorted(set(controls)|scores|{horizon}),controls,scores,horizon

class Config(SimpleNamespace):
    def to_dict(self):return vars(self).copy()

class ClockPlant(CommonPlant):
    def __init__(self,scenario,state,period,external_root=None):
        super().__init__(scenario,state)
        self.config=Config(**vars(self.config));self.score_period=frac(self.config.dt)
        self.horizon=self.score_period*self.config.max_steps;self.config.dt=float(period)
        self.config.max_steps=int(np.ceil(float(self.horizon/frac(period))))
        self.time=Fraction(0);self.anchor_time=self.time;self.anchor=self.positions.copy()
        self.termination='running';self.external_radius=0.;self.original_external=None
        self.external=np.zeros((self.config.max_steps+1,0,2));self.external_velocity=self.external.copy()
        native=state.get('native_static_geometry')
        if native:self._bfc_native_geometry_data=native
        if scenario.get('external_motion'):
            m=scenario['external_motion'];path=Path(external_root)/m['path'];assert sha(path)==m['sha256']
            with np.load(path) as z:self.original_external=z['positions'].copy()
            self.external_radius=m['radius']
            times=[min(i*frac(period),self.horizon) for i in range(self.config.max_steps+1)]
            self.external=np.array([self.external_at(t) for t in times])
            self.external_velocity=np.array([self.external_derivative(t) for t in times])

    def external_at(self,t):
        if self.original_external is None:return np.zeros((0,2))
        q=t/self.score_period;j=min(int(q),len(self.original_external)-1)
        if q==j or j==len(self.original_external)-1:return self.original_external[j].copy()
        return self.original_external[j]+float(q-j)*(self.original_external[j+1]-self.original_external[j])

    def external_derivative(self,t):
        if self.original_external is None:return np.zeros((0,2))
        j=min(int(t/self.score_period),len(self.original_external)-2)
        return (self.original_external[j+1]-self.original_external[j])/float(self.score_period)

    def set_control(self):
        self.anchor=self.positions.copy();self.anchor_time=self.time

    def advance(self,u,end,score):
        c=self.config;before=self.positions.copy();after=self.anchor+float(end-self.anchor_time)*u
        pair=float(point_segment_distance(np.zeros(2),before[self._i]-before[self._j],after[self._i]-after[self._j]).min()-2*c.agent_radius-c.agent_collision_margin) if len(self._i) else np.inf
        wall=float(segment_distance(before[:,None],after[:,None],self.walls[None,:,0],self.walls[None,:,1]).min()-self.instance.geometry.margin) if len(self.walls) else np.inf
        moving=np.inf
        if self.original_external is not None:
            a=before[:,None]-self.external_at(self.time);b=after[:,None]-self.external_at(end)
            moving=float(point_segment_distance(np.zeros(2),a,b).min()-c.agent_radius-self.external_radius)
        hits=dict(agent=pair<=0,static=wall<=0,moving=moving<=0)
        for k,v in hits.items():self.collision_types[k]|=v
        self.collided|=any(hits.values());self.positions=after;self.velocities=u.copy();self.time=end;self.step_count+=1
        success=score and bool(np.all(np.linalg.norm(self.goals-after,axis=1)<=c.goal_tolerance))
        self.termination='collision' if self.collided else 'success' if success else 'timeout' if end==self.horizon else 'running'
        self.done=self.termination!='running'
        return dict(swept_clearance=min(pair,wall,moving),pair=pair,wall=wall,moving=moving,termination=self.termination)
