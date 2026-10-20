"""Frozen A -> tune -> freeze -> fair -> C workflow, no retries."""
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
    if not read(HERE/f'{phase}_MISSING_CASES.json'):return
    while True:
        rows=active();cpus=sum(r['cpus'] for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']);limit=min(18,20-cpus)
        if limit>0:break
        write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase=phase,status='WAITING_RESOURCE'));time.sleep(30)
    cmd=['sbatch',f'--array=0-17%{limit}',str(HERE/'rollout.sbatch'),phase,'18'];response=subprocess.check_output(cmd,text=True);job=re.search(r'(\d+)',response).group(1);write(HERE/f'{phase}_SUBMISSION.json',dict(utc=utc(),command=cmd,response=response,resources_before=rows,cpu_limit=20,gpu_shards_limit=2,preflight_sha256=sha(HERE/f'{phase}_PREFLIGHT.receipt.json')));print(response,flush=True);write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase=phase,status='RUNNING',job=job))
    while True:
        rows=active();live=[r for r in rows if r['job'].split('_')[0]==job];total=sum(r['cpus'] for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']);monitor.append(dict(utc=utc(),phase=phase,active_cpu=total,queue=rows));write(HERE/'RESOURCE_MONITOR.json',monitor);assert total<=20,'Resource cap exceeded, stop new submissions'
        if not live:break
        time.sleep(30)
    assert all((HERE/phase/f'worker_{i}.json').exists() for i in range(18)),('Worker failed; no retry',phase)
def freeze_rhr():
    cases=read(HERE/'tune_CASES.json');db={r['rollout_uid']:r for r in read(HERE/'tune_DB_ROWS.json')};allgrid=read(HERE/'RHR_GRID.json');grids={};selected={}
    for pool in ['students','G']:
        grid=[]
        for k,rule in enumerate(allgrid):
            cs=[c for c in cases if c['arm']==f'RHR_grid{k:02}' and (c['kind']=='student' if pool=='students' else c['kind']=='G')];names=sorted({c['base_controller_name'] for c in cs});assert len(cs)==16*len(names);by={}
            for n in names:
                rr=[db[c['rollout_uid']] for c in cs if c['base_controller_name']==n];by[n]=dict(success=sum(r['success'] and not r['numerical_failure'] for r in rr),unknown=sum(r['numerical_failure'] for r in rr))
            s=sum(r['success'] for r in by.values());u=sum(r['unknown'] for r in by.values());grid.append(dict(index=k,rule=rule,success=s,unknown=u,lower=s/len(cs),upper=(s+u)/len(cs),controllers=by))
        choice=min(grid,key=lambda r:(-r['lower'],r['rule']['speed'],r['rule']['steps'],['always','slow','progress'].index(r['rule']['mode'])));selected[pool]=dict(rule=choice['rule'],grid_index=choice['index'],tuning_lower=choice['lower'],tuning_upper=choice['upper'],selection_ambiguous_with_unknown=any(r['upper']>choice['lower'] for r in grid if r is not choice));grids[pool]=grid
    write(HERE/'RHR_GRID_RESULTS.json',dict(utc=utc(),pools=grids));write(HERE/'RHR_SELECTED.json',dict(utc=utc(),**selected,main_labels_read=False));seal('RHR_FROZEN.receipt.json',[HERE/'RHR_GRID_RESULTS.json',HERE/'RHR_SELECTED.json',HERE/'RHR_GRID.json',HERE/'TUNING_STATES.json',HERE/'fair_rhr.py',HERE/'orchestrate.py'],before_fair_main_labels=True)
def main():
    assert not (HERE/'RUN_STATUS.json').exists()
    for phase in ['A','tune','fair','C']:
        write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase=phase,status='DESIGN'))
        if phase=='fair' and not (HERE/'RHR_FROZEN.receipt.json').exists():
            write(HERE/'fair_GAP.json',dict(utc=utc(),status='GAP',reason='Tuning phase missing; no rule may be selected from main labels'));continue
        try:call('planning.py',phase)
        except subprocess.CalledProcessError as e:
            if e.returncode==86:continue
            raise
        (HERE/'main_protocol.json').write_bytes((HERE/f'{phase}_protocol.json').read_bytes());submit_wait(phase);call('postflight.py',phase)
        if phase=='tune':freeze_rhr()
        if phase=='A':call('analysis.py','A')
        if phase=='fair':call('analysis.py','B')
    call('analysis.py','C');call('finish.py');write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase='delivery',status='COMPLETE'))
if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();print(error,flush=True);write(HERE/'RUN_STATUS.json',dict(utc=utc(),status='STOPPED',error=error));raise
