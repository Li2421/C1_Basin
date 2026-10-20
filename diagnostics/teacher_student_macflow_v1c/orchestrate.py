"""One-shot, frozen-order scheduler; never rerun trials or unknowns."""
from core import *
from shared_rollout_db.src import cache_writer
import subprocess,time,re,traceback
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
monitor=[]
def call(script,*args):
    print('RUN',script,args,flush=True);subprocess.run([PYTHON,'-B',str(HERE/script),*args],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},check=True)
def active():
    lines=subprocess.check_output(['squeue','-r','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%C|%b'],text=True).splitlines();rows=[]
    for line in lines:
        i,n,s,c,g=line.split('|');rows.append(dict(job=i,name=n,state=s,cpus=int(c),gres=g))
    return rows

def submit_wait(phase):
    rows=active();cpus=sum(r['cpus'] for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']);limit=min(18,20-cpus);assert limit>0,'No authorized CPU slot'
    cmd=['sbatch',f'--array=0-17%{limit}',str(HERE/('probe.sbatch' if phase=='probe' else 'rollout.sbatch'))]+(['18'] if phase=='probe' else [phase,'18']);response=subprocess.check_output(cmd,text=True);job=re.search(r'(\d+)',response).group(1);write(HERE/f'{phase}_SUBMISSION.json',dict(utc=utc(),command=cmd,response=response,resources_before=rows,cpu_limit=20,gpu_shards_limit=2,preflight_sha256=sha(HERE/f'{phase}_PREFLIGHT.receipt.json')));print(response,flush=True)
    write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase=phase,status='RUNNING',job=job))
    while True:
        rows=active();live=[r for r in rows if r['job'].split('_')[0]==job];total=sum(r['cpus'] for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']);monitor.append(dict(utc=utc(),phase=phase,active_cpu=total,queue=rows));write(HERE/'RESOURCE_MONITOR.json',monitor);assert total<=20,'External resource changes exceeded total CPU cap; stop further submissions'
        if not live:break
        time.sleep(30)
    if phase=='probe':assert all((HERE/f'imports_probe_{i}.json').exists() for i in range(18))
    else:assert all((HERE/phase/f'worker_{i}.json').exists() for i in range(18)),('Worker failed; no retry',phase)

def freeze_rhr():
    rows=read(HERE/'tune_CASES.json');db={r['rollout_uid']:r for r in read(HERE/'tune_DB_ROWS.json')};names=read(HERE/'REGATE.json')['valid_names'];grid=[]
    for speed in [.10,.15,.25,.35]:
        for trigger in [20,40,80]:
            cases=[c for c in rows if c['rhr_speed']==speed and c['rhr_trigger']==trigger];by={};success=0;unknown=0
            for n in names:
                records=[db[c['rollout_uid']] for c in cases if c['base_controller_name']==n];assert len(records)==16;s=sum(r['success'] and not r['numerical_failure'] for r in records);u=sum(r['numerical_failure'] for r in records);success+=s;unknown+=u;by[n]=dict(success=s,unknown=u,lower=s/16,upper=(s+u)/16)
            grid.append(dict(speed=speed,trigger_steps=trigger,success=success,unknown=unknown,student_mean_lower=success/(len(names)*16),student_mean_upper=(success+unknown)/(len(names)*16),controllers=by))
    choice=min(grid,key=lambda r:(-r['success'],r['speed'],r['trigger_steps']));write(HERE/'RHR_GRID_RESULTS.json',dict(utc=utc(),controllers=names,grid=grid,selection='maximum equal-student mean lower success; ties speed then trigger ascending'));write(HERE/'RHR_SELECTED.json',dict(utc=utc(),speed=choice['speed'],trigger_steps=choice['trigger_steps'],tuning_mean=choice['student_mean_lower'],tuning_unknown=choice['unknown'],evaluation_labels_read=False));seal('RHR_FROZEN.receipt.json',[HERE/'RHR_GRID_RESULTS.json',HERE/'RHR_SELECTED.json',HERE/'TUNING_STATES.json',HERE/'tuned.py',HERE/'orchestrate.py'],before_tuned_main_labels=True)

def main():
    assert not (HERE/'RUN_STATUS.json').exists()
    (HERE/'main_protocol.json').write_bytes((HERE/'pilot_protocol.json').read_bytes());submit_wait('pilot');call('postflight.py','pilot');call('probes.py','design');submit_wait('probe');call('probes.py','freeze')
    journal=cache_writer.append_journal([read(p) for p in sorted((HERE/'probe/raw').glob('*.json'))],'exp_teacher_student_macflow_v1c_probe','unlabeled_provenance');assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals');r=subprocess.run([PYTHON,'-B','-m','shared_rollout_db.src.merger','--experiment','exp_teacher_student_macflow_v1c_probe'],cwd=ROOT,capture_output=True,text=True);(HERE/'probe_MERGE.log').write_text(r.stdout+r.stderr);assert r.returncode==0;write(HERE/'probe_JOURNAL_RECEIPT.json',dict(utc=utc(),journal=str(journal),classification='SUMMARY_ONLY unlabeled probes; no task outcomes'))
    call('freeze_choices.py');call('planning.py','tune');(HERE/'main_protocol.json').write_bytes((HERE/'tune_protocol.json').read_bytes());submit_wait('tune');call('postflight.py','tune');freeze_rhr()
    while not (HERE/'ANALYSIS_READY.receipt.json').exists():time.sleep(10)
    call('planning.py','main');seal('MAIN_IMPLEMENTATION_FROZEN.receipt.json',[HERE/'CHOICES_FROZEN.receipt.json',HERE/'RHR_FROZEN.receipt.json',HERE/'main_PREFLIGHT.receipt.json',HERE/'ANALYSIS_READY.receipt.json',HERE/'run.py',HERE/'tuned.py',HERE/'student.py'],before_main_labels=True);submit_wait('main');call('postflight.py','main');call('analysis.py');call('finish.py');write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='delivery',status='COMPLETE'))
if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();print(error,flush=True);write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='STOPPED',error=error));raise
