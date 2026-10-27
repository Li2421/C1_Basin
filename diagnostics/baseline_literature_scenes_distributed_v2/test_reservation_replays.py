"""Full byte replay before deployment of the new exact search certificate."""
from experiment import *
import concurrent.futures
import multiprocessing
import numpy as np
import time

def init(cpus):os.sched_setaffinity(0,{cpus[os.getpid()%len(cpus)]})

def replay(job):
    if job['kind']=='old':from central_reservation_reference import run
    else:from runtime_reservation_certified import run
    started=time.monotonic();raw=read(job['path']);assert sha(job['path'])==job['sha256'];assert sha(raw['trace_path'])==raw['trace_sha256']
    result,arrays=run(raw['case']);checks={}
    with np.load(raw['trace_path']) as z:
        for key in ('positions','velocities','actions','nominal','safe','goals','swept_clearance'):
            a=np.asarray(arrays[key]);b=z[key];checks[key]=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    terminal={k:result[k]==raw[k] for k in ('success','collision','timeout','numerical_failure','outcome','episode_length')}
    proof=dict(job,passed=all(checks.values()) and all(terminal.values()),arrays=checks,terminal=terminal,seconds=time.monotonic()-started,
        certificates=result.get('reservation_no_goal_certificates',[]),new_research_rollouts=0)
    path=HERE/'reservation_gate_runs'/f"{uid('gate',job)}.json";write(path,proof)
    print(job['kind'],raw['case']['controller_payload']['nominal']['controller'],raw['case']['state']['N'],proof['passed'],round(proof['seconds'],3),flush=True)
    return str(path)

def main():
    assert read(HERE/'RESERVATION_COMPONENT_TESTS.json')['passed'];jobs=[]
    for row in read(HERE/'HISTORICAL_GATE_CASES.json'):
        if row['kind']=='planner_storage':jobs.append(dict(kind='old',path=row['source'],sha256=row['source_sha256']))
    cases=read(HERE/'distributed_baseline_lit_crowded_random_CASES.json');buckets={n:[] for n in (2,4,6,8,10,12,14)}
    for c in cases:
        if c['seed'] or c['controller_payload']['nominal']['type_code']!='prioritized_astar':continue
        path=HERE/'distributed_baseline_lit_crowded_random/raw'/f"{c['rollout_uid']}.json"
        if path.exists():buckets[c['state']['N']].append(path)
    for values in buckets.values():values.sort(key=lambda p:p.name)
    selected=[]
    while len(selected)<16:
        for n in sorted(buckets):
            if buckets[n] and len(selected)<16:selected.append(buckets[n].pop(0))
        assert any(buckets.values()) or len(selected)==16
    jobs += [dict(kind='nowall',path=str(p),sha256=sha(p)) for p in selected];assert len(jobs)==32
    write(HERE/'RESERVATION_GATE_CASES.json',jobs)
    seal('RESERVATION_GATE_CASES.receipt.json',[HERE/'RESERVATION_GATE_CASES.json',HERE/'RESERVATION_CERTIFICATE.md'],before_replays=True,new_science_labels=0)
    from resources import snapshot
    resources=snapshot();workers=min(8,resources['free_cpu_shares']);assert workers>0
    cpus=sorted(os.sched_getaffinity(0))[:workers]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(cpus,),mp_context=multiprocessing.get_context('spawn')) as pool:
        paths=list(pool.map(replay,jobs))
    proofs=[read(p) for p in paths];write(HERE/'RESERVATION_REPLAY_RESULTS.json',proofs);assert all(p['passed'] for p in proofs)
    files=[HERE/n for n in ('reservation_certificate.py','runtime_reservation_certified.py','central_reservation_reference.py','runtime_reservation_certified.py.diff','central_reservation_reference.py.diff','RESERVATION_BACKEND_DERIVATIONS.json','RESERVATION_CERTIFICATE.md','RESERVATION_COMPONENT_TESTS.json','RESERVATION_GATE_CASES.json','RESERVATION_REPLAY_RESULTS.json','test_reservation_components.py','test_reservation_replays.py')]
    seal('RESERVATION_EQUIVALENCE.receipt.json',[*files,*map(Path,paths)],complete_native_unit_cases=81,old_full_replays=16,nowall_full_replays=16,scientific_semantics_unchanged=True,canonical_ids_preserved=True,before_new_backend_research_rollouts=True)
    print('All reservation certificate replay gates passed',flush=True)

if __name__=='__main__':main()
