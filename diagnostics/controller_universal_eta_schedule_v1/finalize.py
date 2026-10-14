from core import *
import subprocess,time
from shared_rollout_db.src.cache_writer import append_journal

def step(script,*args,allow_video_gap=False):
    write(HERE/'FINALIZE_STAGE.json',dict(utc=utc(),stage=script,status='RUNNING'))
    result=subprocess.run([PY,'-B',str(HERE/script),*args],cwd=HERE,text=True,capture_output=True)
    with (HERE/'FINALIZE.log').open('a') as f:f.write(f'\n{utc()} {script} {args}\n{result.stdout}\n{result.stderr}\n')
    if not(allow_video_gap and result.returncode==2 and (HERE/'VIDEO_GATE.json').exists()):result.check_returncode()
    write(HERE/'FINALIZE_STAGE.json',dict(utc=utc(),stage=script,status='VIDEO_INCOMPLETE' if result.returncode else 'PASS',exit_code=result.returncode));return result.returncode

def main():
    prereg();gate();rows=read(HERE/'main_CASES.json');missing=[c['rollout_uid'] for c in rows if not(HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()];assert not missing,('Batch incomplete; no automatic unknown rerun',len(missing))
    # First endpoint opening, deliberately preceding merger/raw label reads.
    opening=HERE/'ENDPOINT_OPENING.receipt.json'
    if opening.exists():
        old=read(opening);assert old['protocol_sha256']==sha(HERE/'protocol.json') and old['case_manifest_sha256']==sha(HERE/'main_CASES.json');assert all(sha(HERE/'main/raw'/f"{x['rollout_uid']}.json")==x['sha256'] for x in old['raw_inventory'])
    else:write(opening,dict(utc=utc(),preregistration_v2_sha256=sha(HERE/'PREREGISTRATION_V2.md'),protocol_sha256=sha(HERE/'protocol.json'),case_manifest_sha256=sha(HERE/'main_CASES.json'),planned_requests=18688,present_raw=len(rows),raw_inventory=[dict(rollout_uid=c['rollout_uid'],sha256=sha(HERE/'main/raw'/f"{c['rollout_uid']}.json")) for c in rows],interim_outcome_analysis=False,main_job=20112,workers_finished_dependency=True))
    for start in range(0,len(rows),128):append_journal([read(HERE/'main/raw'/f"{c['rollout_uid']}.json") for c in rows[start:start+128]],EXP,'complete_raw_recovery')
    step('asset_audit.py');step('merge.py');step('postflight.py','main');step('analyze.py');video_exit=step('videos.py',allow_video_gap=True);step('report.py')
    write(HERE/'NUMERICAL_COMPLETE.receipt.json',dict(utc=utc(),status='NUMERICAL_COMPLETE',main=18688,gate=32,postflight_sha256=sha(HERE/'main_POSTFLIGHT.json'),results_sha256=sha(HERE/'results.json'),report_sha256=sha(HERE/'REPORT.md'),handoff_sha256=sha(HERE/'HANDOFF.md'),video_complete=video_exit==0,video_exit_code=video_exit))
    if video_exit:return video_exit
    write(HERE/'COMPLETE.json',dict(utc=utc(),status='COMPLETE',main=18688,gate=32,video_manifest_sha256=sha(HERE/'videos/manifest.json')));return 0
if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        write(HERE/'FINALIZE_ERROR.json',dict(utc=utc(),type=type(exc).__name__,message=str(exc),complete=False));raise
