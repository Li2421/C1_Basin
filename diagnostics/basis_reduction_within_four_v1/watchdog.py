"""Recover infrastructure coordinator exits; never bypass a hard stop."""
from pathlib import Path
import subprocess,os,time,json,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];restarts=0
while True:
 time.sleep(15)
 if(HERE/'COMPLETE.json').exists():break
 rows=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%j'],text=True).splitlines()
 if any(name=='brd1_pipeline'for name in rows):continue
 if(HERE/'PROTECTION_VIOLATION.jsonl').exists():break
 error=HERE/'PIPELINE_ERROR.json'
 if error.exists()and'HARD_STOP'in error.read_text():break
 if restarts>=3:break
 restarts+=1
 with(HERE/'ITERATION_LOG.md').open('a')as f:f.write('\n'+datetime.datetime.now().astimezone().isoformat()+f'：协调作业异常退出，尝试恢复{restarts}/3；保持原UID、随机流与pending frontier。\n')
 with(HERE/'watchdog_resume.log').open('a')as log:
  subprocess.Popen(['bash',str(HERE/'resume.sh')],cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
