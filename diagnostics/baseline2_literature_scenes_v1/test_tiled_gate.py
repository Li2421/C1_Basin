"""Path-byte checks plus sixteen saved complete planner-episode replays."""
from experiment import *
import numpy as np
import time
from types import SimpleNamespace


def paths():
    from certified_planning import Grid as Old
    from tiled_planning import Grid as New
    box=np.asarray([[[-2.,-2.],[2.,-2.]],[[2.,-2.],[2.,2.]],
                    [[2.,2.],[-2.,2.]],[[-2.,2.],[-2.,-2.]]])
    env=SimpleNamespace(walls=box,config=SimpleNamespace(agent_radius=.1,agent_collision_margin=0.),
                        instance=SimpleNamespace(geometry=SimpleNamespace(margin=.1)))
    rng=np.random.default_rng(2026102010);records=[]
    for i in range(32):
        old,new=Old(env,.1,.1),New(env,.1,.1)
        old.max_time=new.max_time=100
        start=rng.uniform(-1.5,1.5,2);goal=rng.uniform(-1.5,1.5,2)
        negative=[] if i%2 else [(30,1.,0.,1.,0.)]
        if i%4==0:start=np.array([0.,0.]);goal=np.array([1.,0.])
        if i%4==1:
            for grid in (old,new):grid.reserved=[np.asarray([[0.,-.5],[0.,0.],[0.,.5]])]
        outputs=[];seconds=[]
        for grid in (old,new):
            t=time.monotonic();outputs.append(grid.search(start,goal,negative if i%2==0 else None));seconds.append(time.monotonic()-t)
        a,b=outputs
        assert (a is None)==(b is None),i
        if a is not None:assert a.dtype==b.dtype and a.tobytes()==b.tobytes(),i
        records.append(dict(index=i,bytes_equal=True,canonical_seconds=seconds[0],tiled_seconds=seconds[1]))
    write(HERE/'TILED_PATH_QUERY_CHECK.json',records)


def replay():
    from flow_runtime_tiled import run
    pool={};five={}
    for stage in ('capability_lit_circle_five','safety_lit_circle_five'):
        for path in (HERE/stage/'raw').glob('*.json'):
            raw=read(path);c=raw['case'];ctl=c['controller_payload']['nominal']
            if ctl['type_code'] not in ('prioritized_astar','conflict_based_search') or c['seed']!=0 or raw['numerical_failure']:continue
            item=(raw['episode_length'],str(path),raw)
            pool.setdefault(ctl['controller'],[]).append(item)
            if c['state']['N']==5:five.setdefault(ctl['controller'],[]).append(item)
    assert len(pool)==12
    preferred={'prioritized_astar0','prioritized_astar3','conflict_based_search0','conflict_based_search3'}
    selected=[]
    for name in sorted(pool):
        candidates=five.get(name,[]) if name in preferred else pool[name]
        assert candidates,name
        selected.append(sorted(candidates)[0])
    chosen={p for _,p,_ in selected}
    selected+=sorted(x for vv in five.values() for x in vv if x[1] not in chosen)[:4]
    assert len(selected)==16
    records=[]
    for _,path,old in selected:
        rawhash=sha(path);assert sha(old['trace_path'])==old['trace_sha256']
        start=time.monotonic();result,arrays=run(old['case'])
        with np.load(old['trace_path']) as z:
            for key in ('positions','actions','velocities','nominal','safe','cone_residual',
                        'projection_rewrite','swept_clearance'):
                a,b=z[key],arrays[key]
                assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),(path,key)
        for key in ('success','collision','timeout','numerical_failure','outcome','episode_length'):
            assert result[key]==old[key],(path,key)
        assert sha(path)==rawhash and sha(old['trace_path'])==old['trace_sha256']
        records.append(dict(source=path,source_sha256=rawhash,trace=old['trace_path'],
                            trace_sha256=old['trace_sha256'],controller=old['case']['controller_payload']['nominal']['controller'],
                            N=old['case']['state']['N'],first_action_bytes=True,full_trace_bytes=True,
                            termination_and_steps_equal=True,seconds=time.monotonic()-start))
        print(records[-1]['controller'],records[-1]['N'],round(records[-1]['seconds'],3),'exact',flush=True)
    write(HERE/'TILED_PARENT_REPLAY.json',records)
    files=[HERE/n for n in ('tiled_parent_store.hpp','test_tiled_parent_store.cpp',
        'test_tiled_parent_store','build_tiled_backend.py','test_tiled_gate.py',
        'TILED_PARENT_PROOF.json','TILED_PARENT_DICTIONARY_TEST.json',
        'TILED_PATH_QUERY_CHECK.json','TILED_PARENT_REPLAY.json',
        'tiled_planning.py','tiled_nominal.py','flow_runtime_tiled.py',
        'CONTROLLER_INTERIOR_EQUIVALENCE.receipt.json','GRID_CERTIFICATE_EQUIVALENCE.receipt.json')]
    for base in ('astar_kernel','cbs_kernel'):
        files.extend(HERE/f'{base}_{suffix}' for suffix in ('tiled.cpp','tiled.so','TILED_PARENT.diff'))
    seal('TILED_PARENT_EQUIVALENCE.receipt.json',files,
         full_episodes=16,all_twelve_planner_parameters_covered=True,
         no_search_bound_or_algorithm_change=True,no_new_scientific_labels=True)


if __name__=='__main__':paths();replay()
