"""User-authorized 4000-step reporting cutoff; never fabricate a 4800-step label."""
from common import *
from schedule import merge,log
import pickle,numpy as np,subprocess

def main():
    a=read(HERE/'USER_CUTOFF_AMENDMENT.json');rid=a['rollout_uid'];c=a['case'];cutoff=a['cutoff_step']
    assert not(HERE/'main/raw'/f'{rid}.json').exists()
    pid=a['owner_before_pause']['pid'];assert not Path(f'/proc/{pid}').exists()
    for p,h in read(HERE/'CHECKPOINT_COMPATIBILITY.json')['implementation_sources'].items():assert sha(p)==h
    receipts=[]
    for p in(HERE/'operational_checkpoints'/rid).glob('*.receipt.json'):
        r=read(p);receipts.append((r['next_step'],str(p),r))
    step,path,receipt=max(receipts,key=lambda r:(r[0],r[1]));assert step>=cutoff
    assert receipt['scientific_label']is None and receipt['case_hash']==uid('checkpoint_case',c)and sha(receipt['path'])==receipt['sha256']
    with Path(receipt['path']).open('rb')as f:state=pickle.load(f)
    arrays={field:np.asarray(state[field])[:cutoff+1 if field in('positions','velocities')else cutoff]for field in('positions','velocities','nominal','safe','actions')}
    arrays['goals']=np.asarray(state['env'].goals);folder=HERE/'user_cutoff';folder.mkdir(exist_ok=True);trace=folder/'prefix_4000.npz';np.savez_compressed(trace,**arrays)
    write(HERE/'USER_CUTOFF_EVIDENCE.json',dict(utc=utc(),rollout_uid=rid,status='UNCOMPLETED_USER_CUTOFF',reported_steps=cutoff,safe_pause_step=step,
        original_horizon=a['original_horizon'],scientific_outcome=None,original_horizon_success=None,original_horizon_timeout=None,deadlock=None,
        checkpoint_receipt=path,checkpoint_receipt_sha256=sha(path),checkpoint_path=receipt['path'],checkpoint_sha256=receipt['sha256'],
        partial_trace_path=str(trace),partial_trace_sha256=sha(trace),actual_partial_episode_wall_seconds=float(state['elapsed_before']),
        prefix_step_seconds_sum=float(np.sum(state['durations'][:cutoff])),extra_simulated_steps_not_used_for_cutoff_report=step-cutoff,
        no_partial_rollout_database_row=True,no_partial_scientific_journal=True,primary_and_G1_G2_G3_unaffected=True))
    cases=read(HERE/'main_MISSING_CASES.json');missing=[v for v in cases if not(HERE/'main/raw'/f"{v['rollout_uid']}.json").exists()]
    assert len(missing)==1 and missing[0]['rollout_uid']==rid
    for v in cases:
        if v['rollout_uid']!=rid:assert(HERE/'main/journal_receipts'/f"{v['rollout_uid']}.json").exists()
    write(HERE/'main_GAP.json',dict(utc=utc(),reason='USER_AUTHORIZED_4000_STEP_CUTOFF',uncompleted_requests=[rid],not_a_timeout_or_deadlock=True,original_manifest_unchanged=True))
    merge('main');log('completed',stage='main',user_cutoff=True,completed=len(cases)-1,uncompleted=1)
    write(HERE/'main_COMPLETE.json',dict(utc=utc(),stage='main',requests=len(cases),completed=len(cases)-1,infrastructure_retries=0,gaps=True,user_cutoff=True))
    write(HERE/'PROGRESS.json',dict(utc=utc(),stage='statistics_and_delivery',running=0,remaining_execution=0,user_authorized_uncompleted_cells=1))
    subprocess.run([sys.executable,'-B',str(HERE/'statistics_v2_cutoff.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(HERE/'deliver_v2.py')],cwd=ROOT,check=True)

if __name__=='__main__':main()
