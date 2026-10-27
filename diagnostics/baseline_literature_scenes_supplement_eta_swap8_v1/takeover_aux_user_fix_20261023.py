"""Stop only the auxiliary revised-capability worker before regular batch takeover."""
from pathlib import Path
import subprocess,sys,time,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def aux_jobs():
    out=[]
    for line in subprocess.check_output(['squeue','-u','zhihan','-h','-r','-o','%i|%j'],text=True).splitlines():
        jid,name=line.split('|')
        if name=='lit_supp_fix_aux':out.append(jid)
    return out
if __name__=='__main__':
    signaled=[]
    for jid in aux_jobs():
        detail=subprocess.check_output(['scontrol','show','job','-o',jid],text=True);assert str(HERE) in detail
        subprocess.run(['scancel','--signal=USR1',jid],check=True);signaled.append(jid)
    while aux_jobs():time.sleep(5)
    with (HERE/'USER_FIX_AUX_AFTER_GUARD.log').open('w') as log:r=subprocess.run([sys.executable,'-B',str(HERE/'guard_fast.py'),'after_user_fix_aux_before_takeover'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        (HERE/'FATAL_STOP.json').write_text(json.dumps(dict(reason='Auxiliary takeover read-only guard failed')))
        raise RuntimeError('Read-only guard failed')
    folder=HERE/'QUEUES/capability_user_fix_20261023';archive=folder/'aborted_claims';archive.mkdir(exist_ok=True);reopened=[]
    for e in json.loads((folder/'INDEX.json').read_text()):
        claim=folder/'claims'/e['group']
        if not claim.exists() or (claim/'COMPLETE.json').exists():continue
        cs=json.loads(Path(e['path']).read_text())
        if any(not (HERE/'capability_user_fix_20261023/raw'/f"{c['rollout_uid']}.json").exists() for c in cs):claim.rename(archive/(e['group']+'_'+str(time.time_ns())));reopened.append(e['group'])
    (HERE/'USER_FIX_AUX_TAKEOVER.json').write_text(json.dumps(dict(utc=time.time(),aux_jobs_signaled=signaled,old_or_free_workers_signaled=[],abandoned_incomplete_claims_reopened=reopened,unknown_raw_never_retried=True,read_only_passed=True)))
