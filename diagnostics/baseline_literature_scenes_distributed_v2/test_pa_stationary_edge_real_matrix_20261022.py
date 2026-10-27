"""Four-scene saved replans: exact nominal, path bytes, records and certificates."""
from experiment import *
import time
import numpy as np
from result_inventory import find_raw
from states import LiteratureConfig
from flow_scene import LiteratureEnv
from controllers import bound
from reservation_certificate_v2_start_signature_v2_20261022 import ReservationNominal as Reference
from reservation_certificate_v2_stationary_edge_nocache_test_20261022 import ReservationNominal as Candidate
from pa_stationary_edge_stats_20261022 import snapshot


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    os.nice(10)
    rows=[]
    for scene in NAMES:
        stage='distributed_revised_fixed_'+scene
        cases=read(HERE/(stage+'_CASES.json'))
        for ctl in ('prioritized_astar0','prioritized_astar1'):
            subset=[c for c in cases if c['controller_payload']['nominal']['controller']==ctl and c['method']=='fixed_eta10' and c['seed']==0]
            if scene=='lit_crowded_random':subset=[c for c in subset if c['state']['N']==14]
            if scene=='lit_circle_five':subset=[c for c in subset if c['state']['index']==13]
            assert subset,(scene,ctl)
            case=subset[0]
            path=find_raw(case['rollout_uid'],stage,scene);raw=read(path)
            assert sha(raw['trace_path'])==raw['trace_sha256']
            interval=json.loads(case['controller_payload']['nominal']['parameters'])[1]
            with np.load(raw['trace_path']) as z:
                targets=(0,600,1200,1210) if scene=='lit_circle_five' else (0,600)
                steps=sorted({int(min(t,len(z['nominal'])-1)//interval*interval) for t in targets})
                for step in steps:
                    actors=[];answers=[];seconds=[]
                    for cls in (Reference,Candidate):
                        env=LiteratureEnv(case['state'],LiteratureConfig(**case['scenario_payload']['config']))
                        env.positions=z['positions'][step].copy();env.velocities=z['velocities'][step].copy();env.step_count=step
                        actor=cls(case['controller_payload']['nominal']);started=time.monotonic()
                        value=bound(actor.action(env,None),env.config.max_speed).astype(np.float64)
                        value[np.linalg.norm(env.goals-env.positions,axis=-1)<=env.config.goal_tolerance]=0
                        seconds.append(time.monotonic()-started);actors.append(actor);answers.append(value)
                    assert answers[0].tobytes()==answers[1].tobytes()==z['nominal'][step].tobytes(),(scene,ctl,step,'nominal')
                    assert actors[0].planner.planning_records==actors[1].planner.planning_records
                    assert canonical(actors[0].planner.certificates)==canonical(actors[1].planner.certificates),(scene,ctl,step,'certificates')
                    for i,p in actors[0].planner.paths.items():
                        q=actors[1].planner.paths[i]
                        assert (p is None)==(q is None)
                        if p is not None:assert p.tobytes()==q.tobytes(),(scene,ctl,step,i)
                    rows.append(dict(scene=scene,controller=ctl,step=step,source_raw=str(path),raw_sha256=sha(path),trace_sha256=raw['trace_sha256'],all_paths_and_nominal_bytes_equal=True,planning_records_and_certificates_equal=True,reference_seconds=seconds[0],candidate_seconds=seconds[1]))
                    write(HERE/'PA_STATIONARY_EDGE_REAL_MATRIX_PROGRESS_20261022.json',dict(utc=utc(),queries=rows,cache_stats=snapshot()))
                    print('PASS PA saved query',scene,ctl,step,seconds,flush=True)
    target=HERE/'PA_STATIONARY_EDGE_REAL_MATRIX_20261022.json';assert not target.exists()
    write(target,dict(utc=utc(),passed=True,saved_queries=rows,cache_stats=snapshot(),environment_steps=0,new_rollouts=0))
    print('Complete',len(rows),'saved queries',snapshot(),flush=True)


if __name__=='__main__':main()
