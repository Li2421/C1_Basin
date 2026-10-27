"""Historical full-trajectory replay gate; never adds scientific labels."""
from experiment import *
import concurrent.futures
import multiprocessing
import time

def initialize(cpus):
    os.sched_setaffinity(0,{cpus[os.getpid()%len(cpus)]})

def replay(job):
    import numpy as np
    if job['engine']=='distributed_N1':from runtime import run
    else:from central_tiled_runtime import run
    started=time.monotonic();old=read(job['source'])
    assert sha(job['source'])==job['source_sha256']
    assert sha(job['trace'])==job['trace_sha256']
    result,arrays=run(old['case'])
    comparisons={}
    with np.load(job['trace']) as source:
        for key in ('positions','velocities','actions','nominal','safe','goals','swept_clearance'):
            a=np.asarray(arrays[key]);b=source[key]
            comparisons[key]=dict(byte_equal=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),
                replay_sha256=hashlib.sha256(a.tobytes()).hexdigest(),source_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
                shape=list(a.shape),source_shape=list(b.shape))
            if not comparisons[key]['byte_equal'] and a.shape==b.shape:
                comparisons[key]['max_absolute_error']=float(np.max(abs(a-b))) if a.size else 0.
    fields=('success','collision','timeout','numerical_failure','outcome','episode_length')
    terminal={k:dict(source=old.get(k),replay=result.get(k)) for k in fields}
    passed=all(r['byte_equal'] for r in comparisons.values()) and all(r['source']==r['replay'] for r in terminal.values())
    proof=dict(job,utc=utc(),passed=passed,comparisons=comparisons,terminal=terminal,
        first_action_byte_equal=comparisons['actions']['byte_equal'],seconds=time.monotonic()-started,
        error=result.get('error'),new_research_labels=0)
    path=HERE/'gate_runs/attempt1'/f"{uid('gate',dict(engine=job['engine'],source=job['source']))}.json"
    write(path,proof)
    print(job['engine'],','.join(job['roles']),old['case']['controller_payload']['nominal']['controller'],
          old['case']['state']['scene'],passed,round(proof['seconds'],2),flush=True)
    return str(path)

def main():
    prereg();assert read(HERE/'UNIT_TESTS.json')['passed']
    from resources import snapshot
    resources=snapshot();assert not resources['jobs'],resources
    gates=read(HERE/'HISTORICAL_GATE_CASES.json');jobs={}
    for row in gates:
        engine='distributed_N1' if row['kind']=='capability_reuse' else 'central_tiled'
        key=(engine,row['source'])
        if key not in jobs:jobs[key]=dict(row,engine=engine,roles=[])
        jobs[key]['roles'].append(row['kind'])
    cpus=sorted(os.sched_getaffinity(0))[:8];paths=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(cpus),initializer=initialize,
            initargs=(cpus,),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(replay,job) for job in jobs.values()]
        for future in concurrent.futures.as_completed(futures):paths.append(future.result())
    proofs=[read(p) for p in paths]
    write(HERE/'HISTORICAL_REPLAY_RESULTS.json',proofs)
    assert all(r['passed'] for r in proofs),'Inspect preserved gate_runs/attempt1 differences; no new study rollouts authorized until fixed.'
    caps=[r for r in proofs if 'capability_reuse' in r['roles']]
    assert len(caps)==24
    seal('CAPABILITY_REUSE.receipt.json',[HERE/'HISTORICAL_GATE_CASES.json',HERE/'HISTORICAL_REPLAY_RESULTS.json',*[Path(p) for p in paths if 'capability_reuse' in read(p)['roles']]],
         per_scene_replays=8,total_replays=24,N1_full_positions_and_first_action_byte_equal=True,old_qualification_reused=True)
    seal('HISTORICAL_EXECUTION_GATE.receipt.json',[HERE/'HISTORICAL_REPLAY_RESULTS.json',HERE/'test_replays.py',*[Path(p) for p in paths]],
         logical_checks=len(gates),unique_physical_replays=len(jobs),new_scientific_labels=0)
    print('All historical replay checks passed',len(gates),'logical checks',len(jobs),'unique replays',flush=True)

if __name__=='__main__':main()
