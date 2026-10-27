"""Full byte replay before deployment of the new exact search certificate."""
from experiment import *
import concurrent.futures
import multiprocessing
import numpy as np
import time

def init(cpus):os.sched_setaffinity(0,{cpus[os.getpid()%len(cpus)]})

def replay(job):
    if job['kind']=='old':from central_reservation_reference_v2 import run
    else:from runtime_reservation_v2 import run
    started=time.monotonic();raw=read(job['path']);assert sha(job['path'])==job['sha256'];assert sha(raw['trace_path'])==raw['trace_sha256']
    result,arrays=run(raw['case']);checks={}
    with np.load(raw['trace_path']) as z:
        for key in ('positions','velocities','actions','nominal','safe','goals','swept_clearance'):
            a=np.asarray(arrays[key]);b=z[key];checks[key]=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    terminal={k:result[k]==raw[k] for k in ('success','collision','timeout','numerical_failure','outcome','episode_length')}
    proof=dict(job,passed=all(checks.values()) and all(terminal.values()),arrays=checks,terminal=terminal,seconds=time.monotonic()-started,
        certificates=result.get('reservation_no_goal_certificates',[]),new_research_rollouts=0)
    path=HERE/'reservation_v2_gate_runs'/f"{uid('gate',job)}.json";write(path,proof)
    print(job['kind'],raw['case']['controller_payload']['nominal']['controller'],raw['case']['state']['N'],proof['passed'],round(proof['seconds'],3),flush=True)
    return str(path)

def main():
    assert read(HERE/"RESERVATION_V2_COMPONENT_TESTS.json")["passed"];jobs=read(HERE/"RESERVATION_GATE_CASES.json")
    write(HERE/'RESERVATION_V2_GATE_CASES.json',jobs)
    seal('RESERVATION_V2_GATE_CASES.receipt.json',[HERE/'RESERVATION_V2_GATE_CASES.json',HERE/'RESERVATION_CERTIFICATE_V2.md'],before_replays=True,new_science_labels=0)
    from resources import snapshot
    resources=snapshot();workers=min(8,resources["cpu_limit"]-resources["scheduled_cpus"]);assert workers>0
    cpus=sorted(os.sched_getaffinity(0))[:workers]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(cpus,),mp_context=multiprocessing.get_context('spawn')) as pool:
        paths=list(pool.map(replay,jobs))
    proofs=[read(p) for p in paths];write(HERE/'RESERVATION_V2_REPLAY_RESULTS.json',proofs);assert all(p['passed'] for p in proofs)
    files=[HERE/"reservation_certificate.py",HERE/"RESERVATION_EQUIVALENCE.receipt.json"]+[HERE/n for n in ('reservation_certificate_v2.py','runtime_reservation_v2.py','central_reservation_reference_v2.py','runtime_reservation_v2.py.diff','central_reservation_reference_v2.py.diff','RESERVATION_V2_BACKEND_DERIVATIONS.json','RESERVATION_CERTIFICATE_V2.md','RESERVATION_V2_COMPONENT_TESTS.json','RESERVATION_V2_GATE_CASES.json','RESERVATION_V2_REPLAY_RESULTS.json','test_reservation_v2_components.py','test_reservation_v2_replays.py')]
    seal('RESERVATION_V2_EQUIVALENCE.receipt.json',[*files,*map(Path,paths)],complete_native_unit_cases=82,old_full_replays=16,nowall_full_replays=16,scientific_semantics_unchanged=True,canonical_ids_preserved=True,before_new_backend_research_rollouts=True)
    print('All reservation certificate replay gates passed',flush=True)

if __name__=='__main__':main()
