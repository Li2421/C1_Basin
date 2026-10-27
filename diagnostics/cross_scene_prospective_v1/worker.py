"""One CPU worker: deadline snapshots, immutable results and official journals."""
from common import *
import signal,numpy as np

STOP=False
PAUSE=False

def stop(*_):
    global STOP,PAUSE
    STOP=True;PAUSE=True

def save(stage,c,result,arrays):
    trace=HERE/stage/'traces'/f"{c['rollout_uid']}.npz";trace.parent.mkdir(parents=True,exist_ok=True)
    tmp=trace.with_suffix('.partial.npz');np.savez_compressed(tmp,**arrays)
    with tmp.open('rb')as f:os.fsync(f.fileno())
    tmp.replace(trace);r=record(c,dict(result,trace_path=str(trace),trace_sha256=sha(trace)))
    raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json";write(raw,r)
    j=cache_writer.append_journal([r],EXP,worker_id=f'{stage}_{os.getpid()}')
    write(HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json",dict(journal=str(j),raw_sha256=sha(raw)))
    return r

def alias(stage,c):
    rid=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))))
    path=HERE/stage/'raw'/f'{rid}.json';old=read(path)
    assert sha(old['trace_path'])==old['trace_sha256']
    omitted=('case','rollout_uid','state_uid','controller_uid','scenario_uid','eta_uid','seed_key','timestamp','eta',
        'future_index','initial_positions','initial_velocities','goals','cache_identity_schema')
    r={k:v for k,v in old.items()if k not in omitted}
    r.update(deterministic_source=dict(path=str(path),sha256=sha(path),rollout_uid=rid,
        reason='Unchanged historical deterministic planner/gradient controller ignores the PRNG; same input, geometry and safety chain.'),elapsed_seconds=0.,new_physical_steps=0)
    r=record(c,r);raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json";write(raw,r)
    j=cache_writer.append_journal([r],EXP,worker_id=f'{stage}_alias_{os.getpid()}')
    write(HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json",dict(journal=str(j),raw_sha256=sha(raw)))
    return r

def main(stage):
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    assert read(HERE/'REPLAY_CHECK.json')['passed'] and read(HERE/'EXECUTION_CHECK.json')['passed']
    for receipt in ('MODEL_FROZEN.receipt.json','EXECUTION_FROZEN.receipt.json'):
        for p,h in read(HERE/receipt)['files'].items():assert sha(p)==h,p
    verify_inputs()
    folder=HERE/'QUEUES'/stage;receipt=read(folder/'QUEUE.receipt.json')
    assert sha(folder/'INDEX.json')==receipt['index_sha256']
    limit=float(os.environ.get('XSP_TIME_LIMIT_S','10800'));end=time.monotonic()+limit;deadline=end-3300
    try:
        for entry in read(folder/'INDEX.json'):
            if STOP or time.monotonic()>deadline or (HERE/'STOP.receipt.json').exists():return
            done=folder/'done'/entry['group'];claim=folder/'claims'/entry['group']
            if done.exists():continue
            claim.parent.mkdir(parents=True,exist_ok=True)
            try:claim.mkdir()
            except FileExistsError:continue
            job=(os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID'])if 'SLURM_ARRAY_JOB_ID'in os.environ else os.environ.get('SLURM_JOB_ID')
            write(claim/'OWNER.json',dict(job=job,pid=os.getpid(),utc=utc(),stage=stage))
            assert sha(entry['path'])==entry['sha256']
            for c in read(entry['path']):
                if STOP:return
                raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json";marker=HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json"
                if raw.exists():
                    r=read(raw);assert sha(r['trace_path'])==r['trace_sha256']
                    if not marker.exists():
                        j=cache_writer.append_journal([r],EXP,worker_id=f'{stage}_repair_{os.getpid()}');write(marker,dict(journal=str(j)))
                    continue
                write(HERE/'worker_status'/f'{job}.json',dict(utc=utc(),stage=stage,scene=c['prospective_scene'],
                    controller=c['controller_payload']['nominal']['controller'],rollout_uid=c['rollout_uid'],state=c['state']['index']))
                kind=c['controller_payload']['nominal']['type_code']
                if c['seed']==1 and kind in ('gradient_mpc','prioritized_astar','conflict_based_search'):
                    r=alias(stage,c)
                else:
                    if c.get('cross_scene_new'):
                        from runtime_prospective_checkpointed import run
                        from prospective_checkpoint_store import CheckpointStore,CheckpointPause
                        try:r,a=run(c,CheckpointStore(c,pause=lambda:PAUSE or time.monotonic()>end-120))
                        except CheckpointPause as e:
                            print('Operational checkpoint, no label:',c['rollout_uid'],str(e),flush=True);return
                    else:
                        from execution import execute
                        r,a=execute(c)
                    r=save(stage,c,r,a)
                print(stage,c['prospective_scene'],c['controller_payload']['nominal']['controller'],c['state']['index'],
                    c['requested_action'],c['seed'],r['outcome'],round(r['elapsed_seconds'],3),flush=True)
            write(done,dict(utc=utc(),count=entry['count']))
    except BaseException as e:
        write(HERE/'WORKER_FATAL'/f'{os.getpid()}.json',dict(utc=utc(),type=type(e).__name__,message=str(e),
            traceback=__import__('traceback').format_exc()))
        raise

if __name__=='__main__':main(sys.argv[1])
