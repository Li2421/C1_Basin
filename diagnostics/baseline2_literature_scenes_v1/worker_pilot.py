from experiment import *
from registry import save,save_alias
import time

def main(stage,shard,shards):
    prereg();proof=read(HERE/'PROJECTION_MEMO_EQUIVALENCE.receipt.json');assert all(sha(p)==h for p,h in proof['files'].items());cases=read(HERE/f'{stage}_MISSING_CASES.json');from pilot_runtime import run
    count=0;start=time.monotonic()
    if shards==0:
        groups={}
        for c in cases:
            group=uid('group',[c['state_uid'],c['controller_uid'],c['eta_uid']]);groups.setdefault(group,[]).append(c)
        assigned=[];claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True)
        def queue():
            for group,values in groups.items():
                if all((HERE/stage/'raw'/f"{c['rollout_uid']}.json").exists() for c in values):continue
                claim=claims/group
                try:claim.mkdir()
                except FileExistsError:continue
                write(claim/'OWNER.json',dict(utc=utc(),job=os.environ.get('SLURM_JOB_ID'),pid=os.getpid(),rollout_uids=[c['rollout_uid'] for c in values]))
                for c in sorted(values,key=lambda c:c['seed']):yield c
                write(claim/'COMPLETE.json',dict(utc=utc()))
        assigned=queue()
    else:
        assigned=(c for c in cases if int(hashlib.sha256(canonical([c['state_uid'],c['controller_uid'],c['eta_uid']]).encode()).hexdigest()[:8],16)%shards==shard)
    worker=f"{os.environ.get('SLURM_ARRAY_JOB_ID','local')}_{shard}"
    for c in assigned:
        path=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
        if path.exists():continue
        assert all(sha(HERE/n)==h for n,h in c['controller_payload']['execution_sources'].items())
        kind=c['controller_payload']['nominal']['type_code']
        twin=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))))
        twin_path=HERE/stage/'raw'/f'{twin}.json'
        if c['seed']==1 and kind in ('gradient_mpc','prioritized_astar','conflict_based_search','goal_reference','deterministic') and twin_path.exists():
            result=save_alias(stage,c,twin_path)
        else:
            result,arrays=run(c);result['execution_backend']=dict(canonical='flow_runtime.py',actual='pilot_runtime.py',proof='PROJECTION_MEMO_EQUIVALENCE.receipt.json',source_sha256=sha(HERE/'pilot_runtime.py'));save(stage,c,result,arrays)
        count+=1
        print(c['state']['scene'],c['state']['N'],c['state']['index'],c['controller_payload']['nominal']['controller'],c['method'],c['seed'],result['outcome'],result['elapsed_seconds'],flush=True)
        write(HERE/f'{stage}/worker_{worker}.json',dict(utc=utc(),completed=count,last=c['rollout_uid'],elapsed=time.monotonic()-start))
    write(HERE/f'{stage}/worker_{worker}_COMPLETE.json',dict(utc=utc(),completed=count))
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
