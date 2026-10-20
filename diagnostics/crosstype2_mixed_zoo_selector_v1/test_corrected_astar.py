from common import *
from corrected_astar_reference import StraightnessCheckedGrid,CorrectedPrioritizedAStar
from spacetime_astar import SpaceTimeGrid,segment_distance
from types import SimpleNamespace


class PythonStraightnessGrid(SpaceTimeGrid):
    def search(self,start,goal):
        original=super().search(start,goal)
        if original is None:return None
        self.reserved.pop();candidate=np.linspace(np.asarray(start,float),np.asarray(goal,float),len(original))
        valid=all(self.valid_edge(candidate[t],candidate[t+1],t) for t in range(len(candidate)-1))
        path=candidate if valid else original;self.reserved.append(path);return path


def main():
    assert read(HERE/'FAST_ASTAR_EQUIVALENCE.json')['passed']
    checks=[]
    for res in (.05,.1):
        for target in ([1.,0.],[0.,-1.],[1.,.4],[2.,.8],[1.,.7],[.83,-.57]):
            goal=np.asarray(target);path=StraightnessCheckedGrid(res,res/.52,max_time=200).search([0.,0.],goal)
            reference=PythonStraightnessGrid(res,res/.52,max_time=200).search([0.,0.],goal)
            assert path is not None and reference is not None and path.tobytes()==reference.tobytes()
            t=np.clip(path@goal/(goal@goal),0,1);deviation=float(np.linalg.norm(path-t[:,None]*goal,axis=1).max())
            assert deviation<=res+1e-12
            checks.append(dict(resolution=res,goal=target,deviation_cells=deviation/res,accelerated_matches_reference_bitwise=True))
    grid=StraightnessCheckedGrid(.1,.2,separation=.15,max_time=80)
    paths=[grid.search([0.,0.],[1.,0.]),grid.search([.5,-.5],[.5,.5])]
    assert all(p is not None for p in paths)
    minimum=float('inf');same_cell=0
    for t in range(80):
        a,b=paths
        distance=segment_distance(a[min(t,len(a)-1)]-b[min(t,len(b)-1)],a[min(t+1,len(a)-1)]-b[min(t+1,len(b)-1)])
        minimum=min(minimum,distance);assert distance>=.15-1e-12
        same_cell+=int(np.array_equal(np.rint(a[min(t,len(a)-1)]/.1),np.rint(b[min(t,len(b)-1)]/.1)))
    assert same_cell==0
    manifest=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json')['native_manifest']
    checks_replan=[]
    for parameters in read(HERE/'NEW_CONTROLLER_CONFIGS.json')['ASTAR']:
        rt=CorrectedPrioritizedAStar(parameters,manifest)
        positions=np.asarray([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]])
        goals=positions*1.1
        env=SimpleNamespace(positions=positions,goals=goals,velocities=np.zeros((4,2)),config=rt.config,step_count=0)
        for t in range(53):env.step_count=t;rt.nominal(env,None)
        expected=list(range(0,53,parameters[1]));assert rt.replan_steps==expected
        env2=SimpleNamespace(positions=positions,goals=goals,velocities=np.zeros((4,2)),config=rt.config,step_count=0)
        rt.nominal(env2,None);assert rt.replan_steps==[0]
        checks_replan.append(dict(parameters=parameters,replans=expected,new_episode_resets=True))
    write(HERE/'CORRECTED_ASTAR_TESTS.json',dict(utc=utc(),passed=True,straightness=checks,continuous_priority_separation_minimum=minimum,same_spacetime_cells=same_cell,replanning=checks_replan,old_engineering_evaluation_labels_read=False,physical_rollouts=0))
    print('Corrected A* full unit checks passed: axis/oblique, reservations, replan times, reference equivalence')


if __name__=='__main__':main()
