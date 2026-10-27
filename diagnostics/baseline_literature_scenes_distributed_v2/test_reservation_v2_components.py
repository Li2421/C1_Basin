"""Finite complete native-search tests for the necessary-condition certificate."""
from experiment import *
import numpy as np
from reservation_certificate_v2 import CertifiedGrid,no_terminal_goal as no_goal_vertex
from tiled_planning import Grid
from flow_scene import LiteratureEnv

class UnitGrid(Grid):
    def __init__(self,env,res,quantum):super().__init__(env,res,quantum);self.max_time=6

class UnitCertifiedGrid(CertifiedGrid):
    def __init__(self,env,res,quantum,certificates):super().__init__(env,res,quantum,certificates);self.max_time=6

def main():
    state=dict(scene='lit_crowded_random',N=2,index=0,positions=[[0.,0.],[1.,1.]],goals=[[.6,0.],[0.,1.]],velocities=[[0.,0.],[0.,0.]])
    env=LiteratureEnv(state);rng=np.random.default_rng(2026102009);positive=0;negative=0
    examples=[(np.asarray([0.,0.]),np.asarray([.74,0.]),[np.asarray([[1.06,0.]])])]
    examples.append((np.asarray([0.,.6]),np.asarray([.74,0.]),[np.asarray([[.45,0.]])]))
    for _ in range(80):
        start=rng.integers(-2,3,size=2)*.2;goal=rng.integers(-2,3,size=2)*.2
        paths=[rng.integers(-3,4,size=(rng.integers(1,6),2))*.2 for _ in range(rng.integers(1,4))]
        examples.append((start,goal,paths))
    for start,goal,paths in examples:
        old=UnitGrid(env,.2,.2);new=UnitCertifiedGrid(env,.2,.2,[])
        old.reserved=[p.copy() for p in paths];new.reserved=[p.copy() for p in paths]
        proof=no_goal_vertex(start,goal,paths,.2,6,.3)
        a=old.search(start,goal);b=new.search(start,goal)
        assert (a is None)==(b is None)
        if a is not None:assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        if proof is not None:positive+=1;assert a is None
        else:negative+=1
    assert positive and negative
    write(HERE/'RESERVATION_V2_COMPONENT_TESTS.json',dict(utc=utc(),passed=True,complete_native_small_horizon_comparisons=len(examples),certificate_positive=positive,certificate_negative=negative,production_horizon_unchanged=700,new_research_rollouts=0))
    print('Complete native search certificate tests passed',positive,negative,flush=True)

if __name__=='__main__':main()
