"""Replace only the moving disk row, preserving pair/wall/solver/restoration."""
import bootstrap
from experiment import *
from swap_safety import SwapSafety as OriginalSafety,local_problem as original_problem
from shared_control.hard_projection import HardProjectionConfig
from functools import lru_cache
import numpy as np
GAMMAS=(.05,.1,.15,.2,.25,.3)
V_OBSTACLE=.25
@lru_cache(None)
def selected_gamma():return read(HERE/'MOVING_USER_FIX_UNIT_GATE.json')['selected_gamma']
def local_problem_revision(snapshot,i,gamma=None):
    A,lower,labels=original_problem(snapshot,i)
    for k,label in enumerate(labels):
        if label[0]!='moving':continue
        c=snapshot['config'];p=np.asarray(snapshot['positions']);d=p[i]-np.asarray(snapshot['moving_position']);distance=np.linalg.norm(d)
        if distance<=1e-14:raise RuntimeError('Undefined moving obstacle normal')
        hp=HardProjectionConfig();epsilon=.01*(2*c['agent_radius']+c['agent_collision_margin']+hp.separation_buffer)
        gap=distance-(c['agent_radius']+.3+epsilon);rate=selected_gamma() if gamma is None else gamma
        A[k]=d/distance;lower[k]=V_OBSTACLE-(rate/c['dt'])*gap
    return A,lower,labels
class SwapSafetyRevision(OriginalSafety):
    def __init__(self,snapshot,i,gamma=None):
        self.A,self.lower,self.kinds=local_problem_revision(snapshot,i,gamma);self.max_speed=snapshot['config']['max_speed'];self.config=HardProjectionConfig()
