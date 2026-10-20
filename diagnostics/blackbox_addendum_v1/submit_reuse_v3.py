from common import *
from audit_utils import source_inventory
import subprocess,re

def main(phase):
    prereg();receipt=HERE/f'{phase}_PREFLIGHT.receipt.json';assert receipt.exists();assert not (HERE/f'{phase}_SUBMISSION.json').exists()
    for p,h in read(receipt)['files'].items():assert sha(p)==h,p
    rows=[]
    for line in subprocess.check_output(['squeue','-r','-h','-o','%i|%u|%j|%T|%C|%b'],text=True).splitlines():
        job,user,name,state,cpu,gpu=line.split('|');rows.append(dict(job=job,user=user,name=name,state=state,cpus=int(cpu),gres=gpu))
    used=sum(r['cpus'] for r in rows if r['state'] in ['RUNNING','COMPLETING','CONFIGURING']);limit=min(18,18-used);assert limit>0,'Wait for CPU resource; do not exceed total18'
    cmd=['sbatch',f'--array=0-17%{limit}',str(HERE/'rollout_reuse_v3.sbatch'),phase];response=subprocess.check_output(cmd,text=True);job=re.search(r'(\d+)',response).group(1);write(HERE/f'{phase}_SUBMISSION.json',dict(utc=utc(),command=cmd,response=response,job=job,queue_before=rows,active_cpu_before=used,array_throttle=limit,CPU_cap=18,GPU_cap=2,preflight_sha256=sha(receipt)));write(HERE/'RUN_STATUS.json',dict(utc=utc(),phase=phase,status='RUNNING',job=job));source_inventory(f'imports_submit_{phase}.json');print(response,flush=True)
if __name__=='__main__':main(sys.argv[1])
