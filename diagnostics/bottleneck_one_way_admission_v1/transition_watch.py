"""User-authorized handoff after v3 rescue; never edit v3 files."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'diagnostics/scene_types_oracle_coverage_v3'

def run():
    while True:
        p=OLD/'PRIORITY_RESCUE_SUMMARY.json'
        if p.exists() and json.loads(p.read_text()).get('complete'):break
        time.sleep(1)
    # Explicit latest user instruction authorizes stopping this old task only
    # after rescue completes. It is not resource preemption by the new study.
    rows=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j'],text=True)
    cancelled=[]
    for line in rows.splitlines():
        job,name=line.split('|')
        if name.startswith('stb3_'):
            subprocess.run(['scancel',job],check=True);cancelled.append(job)
    # Disable old auto-recovery, if still alive, to honor the requested pause.
    for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines():
        bits=line.strip().split(None,1)
        if len(bits)==2 and bits[1].startswith('/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python ') and any(str(OLD/n) in bits[1] or 'diagnostics/scene_types_oracle_coverage_v3/'+n in bits[1] for n in ['finish_when_ready.py','resource_watch.py']):
            try:os.kill(int(bits[0]),15)
            except ProcessLookupError:pass
    records=[]
    for name in ['PRIORITY_RESCUE_SUMMARY.json','PRIORITY_RESCUE_STATE.json','SEARCH_STATE.json','REPRESENTATIVE17_COMPARISON.json']:
        data=(OLD/name).read_bytes();dest=HERE/'inputs'/'final_v3'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
        records.append(dict(source=str(OLD/name),copy=str(dest),sha256=hashlib.sha256(data).hexdigest()))
    summary=json.loads((HERE/'inputs/final_v3/PRIORITY_RESCUE_SUMMARY.json').read_text())
    state=json.loads((HERE/'inputs/final_v3/PRIORITY_RESCUE_STATE.json').read_text())
    known={tuple(x['identity'])for x in summary['witnesses']}
    hard=[g for g in state['targets']if (g['type_id'],g['level'],g['controller'],g['state_index'])not in known]
    (HERE/'HARD_CASES.json').write_text(json.dumps(dict(cases=hard,count=len(hard),source_complete=True,source_sha256=records[0]['sha256']),indent=2)+'\n')
    (HERE/'HANDOFF_COMPLETE.json').write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),cancelled_old_jobs=cancelled,authorization='User: 跑完这个就先停下来吧，跑接下来这个',old_files_modified=False,snapshots=records,hard_cases=len(hard)),indent=2)+'\n')
    while not (HERE/'READY_TO_RUN').exists():time.sleep(10)
    for retry in range(4):
        result=subprocess.run(['bash',str(HERE/'resume.sh')],cwd=ROOT)
        if result.returncode==0:return
        error=HERE/'PIPELINE_ERROR.json'
        if error.exists()and 'HARD_STOP'in error.read_text():return
        with(HERE/'ITERATION_LOG.md').open('a')as f:f.write(f'\n新任务协调进程异常退出，基础设施自动恢复 {retry+1}/3。\n')
        if retry<3:time.sleep(30)

if __name__=='__main__':run()
