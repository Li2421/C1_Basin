"""One CPU, fixed deadline; workers only write local artifacts and official journals."""
from common import *
from execution import execute
import numpy as np
import signal

STOP=False
PAUSE=False


def stop(signum,*_):
    global STOP,PAUSE
    STOP=True
    PAUSE=True


def save(stage,c,result,arrays):
    folder=HERE/stage;trace=folder/'traces'/f"{c['rollout_uid']}.npz"
    trace.parent.mkdir(parents=True,exist_ok=True);tmp=trace.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**arrays)
    with tmp.open('rb')as f:os.fsync(f.fileno())
    tmp.replace(trace)
    raw=folder/'raw'/f"{c['rollout_uid']}.json"
    result=record(c,dict(result,trace_path=str(trace),trace_sha256=sha(trace)))
    write(raw,result)
    journal=cache_writer.append_journal([result],EXP,worker_id=f'{stage}_{os.getpid()}')
    write(folder/'journal_receipts'/f"{c['rollout_uid']}.json",dict(journal=str(journal),raw_sha256=sha(raw)))
    return result


def deterministic_alias(stage,c):
    from shared_rollout_db.src.rollout_db import uid as exact_uid
    sk=canonical(dict(future_index=0))
    rid=exact_uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=sk))
    source=HERE/stage/'raw'/f'{rid}.json';old=read(source)
    assert sha(old['trace_path'])==old['trace_sha256']
    omitted=('case','rollout_uid','state_uid','controller_uid','scenario_uid','eta_uid','seed_key','timestamp','eta',
             'future_index','initial_positions','initial_velocities','goals','cache_identity_schema')
    result={k:v for k,v in old.items()if k not in omitted}
    result.update(deterministic_source=dict(path=str(source),sha256=sha(source),rollout_uid=rid,
        reason='Frozen deterministic controller ignores PRNG; same input and geometry, same projection chain.'),
        elapsed_seconds=0.,new_physical_steps=0)
    result=record(c,result);raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json";write(raw,result)
    j=cache_writer.append_journal([result],EXP,worker_id=f'{stage}_alias_{os.getpid()}')
    write(HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json",dict(journal=str(j),raw_sha256=sha(raw)))
    return result


def main(stage):
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    from heartbeat import start as start_heartbeat
    start_heartbeat()
    assert read(HERE/'REPLAY_CHECK.json')['passed']
    assert read(HERE/'EXTENSION_UNIT_TESTS.json')['passed']
    verify_inputs()
    assert sha(HERE/'native_resume/cbs_kernel_resume.so')==read(HERE/'BACKEND_BUILD.json')['native_sha256']
    folder=HERE/'QUEUES'/stage;receipt=read(folder/'QUEUE.receipt.json')
    assert sha(folder/'INDEX.json')==receipt['index_sha256']
    limit=float(os.environ.get('CROSS_SCENE_TIME_LIMIT_S','10800'));end=time.monotonic()+limit;deadline=end-3300
    for entry in read(folder/'INDEX.json'):
        if STOP or time.monotonic()>deadline or (HERE/'FATAL_STOP.json').exists():return
        required=min(entry.get('budget_seconds',entry['estimated_seconds']),max(1.,limit-120))
        if time.monotonic()+required>end-60:continue
        done=folder/'done'/entry['group'];claim=folder/'claims'/entry['group']
        if done.exists():continue
        claim.parent.mkdir(parents=True,exist_ok=True)
        try:claim.mkdir()
        except FileExistsError:continue
        job=(os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID'])if 'SLURM_ARRAY_JOB_ID'in os.environ else os.environ.get('SLURM_JOB_ID')
        write(claim/'OWNER.json',dict(pid=os.getpid(),job=job,utc=utc(),stage=stage))
        assert sha(entry['path'])==entry['sha256']
        for c in read(entry['path']):
            if STOP:
                # Preserve completed raw and make only the unfinished group available again.
                if claim.exists():
                    for p in claim.iterdir():p.unlink()
                    claim.rmdir()
                return
            raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json";marker=HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json"
            if raw.exists():
                r=read(raw);assert sha(r['trace_path'])==r['trace_sha256']
                if not marker.exists():
                    j=cache_writer.append_journal([r],EXP,worker_id=f'{stage}_repair_{os.getpid()}')
                    write(marker,dict(journal=str(j),recovered_existing_raw=True))
                continue
            if c['seed']==1 and c['controller_payload']['nominal']['type_code']in ('gradient_mpc','prioritized_astar','conflict_based_search'):
                result=deterministic_alias(stage,c)
            else:
                from checkpoint_store import CheckpointStore,CheckpointPause
                from checkpoint_scope import validation as checkpoint_validation
                try:
                    validation=checkpoint_validation(c)
                    if validation is not None:
                        from runtime_checkpointed import run as run_checkpointed
                        result,arrays=run_checkpointed(c,CheckpointStore(c,pause=lambda:PAUSE))
                        result['operational_resume']['validation_receipt']=str(validation)
                        result['operational_resume']['validation_receipt_sha256']=sha(validation)
                    else:
                        result,arrays=execute(c)
                except CheckpointPause as exc:
                    print('operational checkpoint pause; no rollout result',c['rollout_uid'],str(exc),flush=True)
                    return
                result=save(stage,c,result,arrays)
            print(stage,c['state']['scene'],c['controller_payload']['nominal']['controller'],c['state']['index'],
                  c['requested_action'],c['seed'],result['outcome'],round(result['elapsed_seconds'],3),flush=True)
        write(done,dict(utc=utc(),count=entry['count']))


if __name__=='__main__':
    main(sys.argv[1])
