"""No simulation: confirm the shared map-only program supplies free paths."""
import json
from pathlib import Path
import numpy as np
from .runtime import VisibilityNavigation
from diagnostics.physical_state_critic_formal_v1.physical_entities import templates
from diagnostics.controller_four_scene_mechanism_v1.study import read,write,freeze,sha

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'controller_ff_factorial_repair_v1'


def run():
    p=read(SOURCE/'protocol.json');rows=[]
    version=sha(HERE/'runtime.py')[:12]
    freeze(HERE/f'reuse_and_design_audit_{version}.json',dict(
        database_audit='../controller_cross_scene_root_v2/shared_controller_cache_audit.json',
        database_controllers_scanned=232,
        historical_code_checked=['double_bottleneck/expert.py','ring_exchange/expert_v6.py','bottleneck_family/expert.py'],
        reuse='Existing exact geometry distances and FF endpoint controller reused. Existingexperts are scene-specific or simulate plannedtrajectories; not a common deploymentphysicalstate program. No navigation/waypoint/visibility/expert-controller exacteta rolloutlabels found among232registeredcontrollers.',
        scientific_gap='Native LOSO changes both map and learned Flow program; goal-only common controllers supply weak/nonexistent state×eta labels. Need a shared task-capable controller with no learned scene-specific weights before attributing failure solely to geometry transfer.',
        scope='Preparation and label-free geometry audit only. Simulation requires the standard exact-identity cache preflight.',
        new_rollouts=0,code_sha256=sha(HERE/'runtime.py'),source_sha256=sha(SOURCE/'protocol.json'),
        literature=['https://arxiv.org/abs/2204.02877','https://proceedings.mlr.press/v119/raileanu20a.html',
                    'https://www.cs.cmu.edu/afs/cs/academic/class/15456-s14/Handouts/BKOS.pdf']))
    for scene,template in templates().items():
        cfg=p['environments'][scene]['config']['config']
        planner=VisibilityNavigation(template['obstacles'],cfg['agent_radius'],cfg.get('wall_collision_margin',cfg.get('collision_margin',0.)))
        ss=[s for s in p['states'] if s['scenario']==scene]
        missing=0;queries=0;patherrors=0
        for s in ss:
            for start,goal in zip(s['physical']['positions'],s['physical']['goals']):
                pos=np.array(start,float);goal=np.array(goal,float);queries+=1
                for step in range(100):
                    target,ok=planner.waypoint(pos,goal)
                    if not ok:missing+=1;break
                    if not planner.visible(pos,target):patherrors+=1;break
                    if np.linalg.norm(target-goal)<1e-8:break
                    pos=target
                else:patherrors+=1
        rows.append(dict(scene=scene,states=len(ss),agents=queries,roadmap_nodes=len(planner.nodes),unroutable=missing,path_errors=patherrors))
    write(HERE/f'geometry_audit_{version}.json',dict(rows=rows,physical_simulator_steps=0,new_rollouts=0,
        caveat='Map-only polyline feasibility is not joint solvability or closed-loop safety/success. Simulator collision evaluator remains authoritative.'))
    print(rows,flush=True)


if __name__=='__main__':run()
