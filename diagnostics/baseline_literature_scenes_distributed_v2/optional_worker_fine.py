"""Bounded worker for optional versions, with the same official evidence path."""
from experiment import *
from registry import verify_receipt,save,save_alias,repair_journal
import time
import signal
STOP=False

def stop(*args):
    global STOP
    STOP=True

def main(stage):
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    verify_receipt('OPTIONAL_FINE_SCHEDULING.receipt.json');verify_receipt(f'{stage}/FINE_QUEUE.receipt.json')
    prereg();verify_receipt('IMPLEMENTATION_FROZEN.receipt.json');verify_receipt('OPTIONAL_CERTIFIED_BACKENDS_V2.receipt.json');verify_receipt(f'{stage}/WORK_QUEUE.receipt.json');verify_receipt('OPTIONAL_DELIVERY_READY.receipt.json')
    index=read(HERE/stage/'FINE_GROUP_INDEX.json');claims=HERE/stage/'fine_claims';claims.mkdir(parents=True,exist_ok=True);groups=0
    for entry in index:
        if STOP:return
        claim=claims/entry['group']
        if (claim/'COMPLETE.json').exists():continue
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(utc=utc(),job=os.environ['SLURM_JOB_ID'],pid=os.getpid()))
        assert sha(entry['file'])==entry['sha256']
        for c in read(entry['file'])[entry['start']:entry['end']]:
            if STOP:return
            raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if raw.exists():repair_journal(stage,c);continue
            engine=c['execution_engine']
            if engine=='paper_dynamics':
                verify_receipt('DYNAMICS_IMPLEMENTATION.receipt.json')
                from dynamics_runtime_reservation_v2 import run
            elif engine=='local_progress':
                from runtime_local_progress_reservation_v2 import run
            else:
                from runtime_reservation_v2 import run
            import jax
            jax.config.update('jax_enable_x64',False)
            kind=c['controller_payload']['nominal']['type_code']
            twin=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))))
            source=HERE/stage/'raw'/f'{twin}.json'
            if c['seed']==1 and kind in ('gradient_mpc','prioritized_astar','conflict_based_search','goal_reference') and source.exists():result=save_alias(stage,c,source)
            else:
                result,arrays=run(c)
                result['jax_enable_x64']=bool(jax.config.x64_enabled)
                result['optional_implementation_receipt_sha256']=sha(HERE/'OPTIONAL_DELIVERY_READY.receipt.json')
                result['reservation_equivalence_receipt_sha256']=sha(HERE/'RESERVATION_V2_EQUIVALENCE.receipt.json')
                result['execution_backend']=dict(actual='dynamics_runtime_reservation_v2.py' if engine=='paper_dynamics' else 'runtime_local_progress_reservation_v2.py' if engine=='local_progress' else 'runtime_reservation_v2.py')
                if engine=='paper_dynamics' and c['state']['scene'] in ('lit_narrow_passage','lit_crowded_random'):
                    import numpy as np
                    result['acceleration_bound_violation_steps']=int(np.any(np.linalg.norm(arrays['actions'],axis=-1)>1.5+1e-7,axis=-1).sum())
                save(stage,c,result,arrays)
            print(c['state']['scene'],c['controller_payload']['nominal']['controller'],c['method'],c['seed'],result['outcome'],round(result['elapsed_seconds'],3),flush=True)
        write(claim/'COMPLETE.json',dict(utc=utc()));groups+=1
        if groups>=512:os.execv(sys.executable,[sys.executable,'-B',str(HERE/'optional_worker_fine.py'),stage])

if __name__=='__main__':main(sys.argv[1])
