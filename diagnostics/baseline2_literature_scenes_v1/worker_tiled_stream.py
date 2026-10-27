"""Bounded-memory worker; same canonical cases, seeds and certified executor."""
from experiment import *
from registry import save,save_alias
from flow_runtime_tiled_progress import run
import time

def main(stage,worker):
    prereg();proof=read(HERE/'GRID_CERTIFICATE_EQUIVALENCE.receipt.json');assert all(sha(p)==h for p,h in proof['files'].items())
    interior=read(HERE/'CONTROLLER_INTERIOR_EQUIVALENCE.receipt.json');assert all(sha(p)==h for p,h in interior['files'].items())
    for name in ('TILED_PARENT_EQUIVALENCE.receipt.json','TILED_PROGRESS_IO.receipt.json'):
        proof=read(HERE/name);assert all(sha(p)==h for p,h in proof['files'].items())
    queue_receipt=read(HERE/stage/'STREAM_QUEUE.receipt.json');index_path=HERE/stage/'GROUP_INDEX.json';assert queue_receipt['files'][str(index_path)]==sha(index_path);index=read(index_path)
    claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True);count=0;groups=0
    for entry in index:
        if all((HERE/stage/'raw'/f'{rid}.json').exists() for rid in entry['rollout_uids']):continue
        claim=claims/entry['group']
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(utc=utc(),job=os.environ.get('SLURM_JOB_ID'),pid=os.getpid(),rollout_uids=entry['rollout_uids']))
        path=Path(entry['file']);assert path.resolve().is_relative_to(HERE/stage/'queue') and sha(path)==entry['sha256']
        for c in read(path):
            out=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if out.exists():continue
            assert all(sha(HERE/n)==h for n,h in c['controller_payload']['execution_sources'].items())
            kind=c['controller_payload']['nominal']['type_code'];twin=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))));twin_path=HERE/stage/'raw'/f'{twin}.json'
            if c['seed']==1 and kind in ('gradient_mpc','prioritized_astar','conflict_based_search','goal_reference') and twin_path.exists():result=save_alias(stage,c,twin_path)
            else:
                result,arrays=run(c);result['execution_backend']=dict(canonical='flow_runtime.py',actual='flow_runtime_tiled_progress.py',proof='TILED_PROGRESS_IO.receipt.json',source_sha256=sha(HERE/'flow_runtime_tiled_progress.py'));save(stage,c,result,arrays)
            count+=1;print(c['state']['scene'],c['state']['N'],c['state']['index'],c['controller_payload']['nominal']['controller'],c['method'],c['seed'],result['outcome'],round(result['elapsed_seconds'],4),flush=True)
        write(claim/'COMPLETE.json',dict(utc=utc()));groups+=1
        write(HERE/stage/f"worker_stream_{os.environ.get('SLURM_JOB_ID','local')}.json",dict(utc=utc(),completed_in_process=count,groups=groups,last=entry['group']))
        if groups>=128:
            print('Infrastructure: restart clean process after 128 completed groups; no physical case is repeated',flush=True)
            os.execv(sys.executable,[sys.executable,'-B',str(HERE/'worker_tiled_stream.py'),stage,str(worker)])
    write(HERE/stage/f"worker_stream_{os.environ.get('SLURM_JOB_ID','local')}_COMPLETE.json",dict(utc=utc()))
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
