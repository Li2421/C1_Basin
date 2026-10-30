"""Numerical check of published target-geometry assumptions; no cohort filter."""
import guard
from guard import HERE
from scene_catalog import catalog,legal_indices
import json,collections,datetime
import numpy as np


def target_slacks(scene,state,safety_ratio):
    goal=np.asarray(state['goals'],float);wall=np.asarray(scene['walls'],float).reshape(-1,2,2);r=scene['config']['agent_radius']*safety_ratio
    wall_slack=None
    if len(wall):
        a=wall[:,0];edge=wall[:,1]-a;offset=goal[:,None,:]-a
        t=np.clip(np.sum(offset*edge,axis=-1)/np.sum(edge*edge,axis=-1),0,1)
        distance=np.linalg.norm(offset-t[:,:,None]*edge,axis=-1)
        # Common line obstacles are capsules; obstacle surface includes the
        # common wall radius and declared wall collision margin.
        wall_slack=float(distance.min()-scene['config']['wall_radius']-scene['config']['wall_collision_margin']-2*r)
    i,j=np.triu_indices(len(goal),1);pair_slack=float(np.linalg.norm(goal[i]-goal[j],axis=-1).min()-2*r)if len(i)else None
    return dict(goal_wall_min_slack=wall_slack,goal_pair_min_slack=pair_slack,
        evaluated_clauses_satisfied=(wall_slack is None or wall_slack>=0)and(pair_slack is None or pair_slack>0),
        near_numeric_boundary=any(abs(x)<=1e-7 for x in (wall_slack,pair_slack)if x is not None))


def run():
    split=json.loads((HERE/'SPLITS_FROZEN.json').read_text());rows=[]
    for benchmark in ('A','B'):
        scenes=catalog(benchmark);parts=('train','validation','test')if benchmark=='A'else('train','test')
        for part in parts:
            for name in split['benchmark_'+benchmark][part]:
                scene=scenes[name];indices=split['benchmark_'+benchmark]['train_states'if part=='train'else'validation_states'if part=='validation'else'test_states']
                for state in legal_indices(scene,indices):
                    for ratio in (1.,1.1):rows.append(dict(benchmark=benchmark,part=part,scene=name,state=state,
                        category=str(scene['factor']if benchmark=='A'else scene['type_id']),safety_ratio=ratio,
                        **target_slacks(scene,scene['states'][state],ratio)))
    grouped=collections.defaultdict(list)
    for row in rows:grouped[(row['benchmark'],row['part'],row['safety_ratio'])].append(row)
    tables=[dict(benchmark=b,part=p,safety_ratio=r,physical_states=len(v),
        target_geometry_satisfying=sum(x['evaluated_clauses_satisfied']for x in v),
        near_boundary_states=sum(x['near_numeric_boundary']for x in v))for(b,p,r),v in sorted(grouped.items())]
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),rows=rows,tables=tables,new_rollouts=0,
        primary_source='https://arxiv.org/pdf/2503.05848v1',paper_sections='III-A Eq2 defines dsafe=2rsafe;III-E imposes target-to-obstacle distance>=2rsafe and goal-pair separation>2rsafe;V-A uses physical radius0.2m and rsafe0.22m.',
        implementation_note='Pinned author cbf.py uses rsafe for robot-to-obstacle constraints, whereas the prose target-obstacle assumption uses2rsafe; these are recorded separately rather than equated.',
        interpretation='Numerical target-geometry diagnostic only. Failure of these published sufficient assumptions does not prove physical infeasibility or controller failure. No test outcomes used; no primary/applicable-only cohort, parameters or control code changed. Safety-ratio classification is not revised based on low training scores; preplanned author-distance rows remain separate.')
    (HERE/'MGR_DOMAIN_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(run()['tables'],ensure_ascii=False))
