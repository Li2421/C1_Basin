"""One CPU, bounded queue, persisted completed evidence, no SQLite writer."""
from experiment import *
from registry import verify_receipt,save,save_alias,repair_journal
import signal
import time
STOP=False

def stop(*args):
    global STOP
    STOP=True

def main(stage):
    prereg();verify_receipt('IMPLEMENTATION_FROZEN.receipt.json');verify_receipt('RESERVATION_EQUIVALENCE.receipt.json')
    verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')
    index=read(HERE/stage/'GROUP_INDEX.json')
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True);count=0;groups=0
    from runtime_reservation_certified import run as distributed_run
    from central_tiled_runtime import run as historical_run
    for entry in index:
        if STOP:break
        claim=claims/entry['group']
        if (claim/'COMPLETE.json').exists():continue
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(utc=utc(),job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),rollout_uids=entry['rollout_uids']))
        path=Path(entry['file']);assert path.resolve().is_relative_to(HERE/stage/'queue') and sha(path)==entry['sha256']
        for c in read(path):
            if STOP:break
            out=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if out.exists():repair_journal(stage,c);continue
            kind=c['controller_payload']['nominal']['type_code']
            twin=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))))
            source=HERE/stage/'raw'/f'{twin}.json'
            if c['seed']==1 and kind in ('gradient_mpc','prioritized_astar','conflict_based_search','goal_reference') and source.exists():
                result=save_alias(stage,c,source)
            else:
                run=historical_run if c['safety_layer']=='central' and c['state']['scene']!='lit_crowded_random' else distributed_run
                result,arrays=run(c)
                result['execution_backend']=dict(actual='central_tiled_runtime.py' if run is historical_run else 'runtime_reservation_certified.py',implementation_receipt_sha256=sha(HERE/'IMPLEMENTATION_FROZEN.receipt.json'))
                result["reservation_equivalence_receipt_sha256"]=sha(HERE/"RESERVATION_EQUIVALENCE.receipt.json");save(stage,c,result,arrays)
            count+=1
            print(c['controller_payload']['nominal']['controller'],c['state']['N'],c['state']['index'],c['method'],c['seed'],result['outcome'],round(result['elapsed_seconds'],4),flush=True)
        if STOP:break
        write(claim/'COMPLETE.json',dict(utc=utc()));groups+=1
        write(HERE/stage/f"worker_{os.environ['SLURM_JOB_ID']}.json",dict(utc=utc(),completed_in_process=count,groups=groups,last=entry['group']))
        if groups>=128:
            os.execv(sys.executable,[sys.executable,'-B',str(HERE/'worker_reservation_certified.py'),stage])
    write(HERE/stage/f"worker_{os.environ['SLURM_JOB_ID']}_EXIT.json",dict(utc=utc(),graceful_stop=STOP,completed_in_process=count))

if __name__=='__main__':main(sys.argv[1])
