from experiment import *
import subprocess
import fcntl
def _submit(stage,script,cpu,gpu=0):
    prereg();p=HERE/f'{stage}_PREFLIGHT.receipt.json'
    if script in ('train_critic.sbatch','train_critic_extended.sbatch'):p=HERE/f"LEARNING_BUNDLE_{stage.removeprefix('critic_')}.receipt.json"
    if script=='train_pilot.sbatch':p=HERE/f"PILOT_DEMONSTRATIONS_{stage.removeprefix('pilot_train_')}.receipt.json"
    receipt=read(p)
    for f,h in receipt['files'].items():assert sha(f)==h,f
    lines=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%t|%C|%b']).decode()
    (HERE/f'{stage}_SQUEUE_BEFORE.txt').write_text(lines)
    used_cpu=0;used_gpu=0
    for line in lines.splitlines():
        job,state,c,gres=line.split('|')
        # Count every state still present in squeue, including configuring or
        # suspended allocations, as well as expanded pending array children.
        used_cpu+=int(c)
        import re
        for kind,n in re.findall(r'(gpu|shard):(?:[^:,]+:)?(\d+)',gres):used_gpu+=int(n)
    if used_cpu+int(cpu)>16 or used_gpu+int(gpu)>2:
        print('CAPACITY_BUSY; keep two CPU shares for local work',used_cpu,used_gpu,cpu,gpu,flush=True);return
    args=['sbatch','--parsable']
    if script in ('flow.sbatch','flow_memo.sbatch','flow_cached.sbatch','flow_stream.sbatch','flow_certified.sbatch','flow_interior.sbatch','flow_tiled.sbatch','flow_nominal.sbatch','paper_secondary.sbatch','flow_pilot.sbatch','flow_pilot_nominal.sbatch','flow_extended.sbatch'):
        if script in ('flow_memo.sbatch','flow_cached.sbatch','flow_stream.sbatch','flow_certified.sbatch','flow_interior.sbatch','flow_tiled.sbatch','flow_extended.sbatch'):
            name='TILED_PARENT_DEPLOYMENT.receipt.json' if script=='flow_tiled.sbatch' else 'CONTROLLER_INTERIOR_EQUIVALENCE.receipt.json' if script=='flow_interior.sbatch' else 'EXTENDED_EXECUTOR_EQUIVALENCE.receipt.json' if script=='flow_extended.sbatch' else 'GRID_CERTIFICATE_EQUIVALENCE.receipt.json' if script=='flow_certified.sbatch' else 'MEMO_EQUIVALENCE.receipt.json' if script=='flow_memo.sbatch' else 'PROJECTION_MEMO_EQUIVALENCE.receipt.json'
            proof=read(HERE/name);assert all(sha(f)==h for f,h in proof['files'].items())
        args.extend([f'--array=0-{int(cpu)-1}%{int(cpu)}',f'--export=ALL,BASELINE2_STAGE={stage},BASELINE2_SHARDS=0'])
    if script in ('train_critic.sbatch','train_critic_extended.sbatch'):args.extend([f"--export=ALL,BASELINE2_SCENE={stage.removeprefix('critic_')}"])
    if script=='train_pilot.sbatch':args.extend([f"--export=ALL,BASELINE2_SCENE={stage.removeprefix('pilot_train_')}"])
    args.append(str(HERE/script))
    out=subprocess.check_output(args).decode().strip()
    destination=HERE/f'{stage}_SUBMISSION.json'
    if destination.exists():destination=HERE/f'{stage}_RESUBMISSION_{out}.json'
    write(destination,dict(utc=utc(),job_id=out,requested_cpu=cpu,requested_gpu=gpu,previous_squeue=lines,script_sha256=sha(HERE/script),preflight_sha256=sha(p)))
    print(out)
def main(stage,script,cpu,gpu=0):
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _submit(stage,script,cpu,gpu)
if __name__=='__main__':main(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 0)
