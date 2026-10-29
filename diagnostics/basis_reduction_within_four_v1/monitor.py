"""Read-only progress observation; never controls scientific labels."""
from pathlib import Path
import json,subprocess,datetime
HERE=Path(__file__).resolve().parent

def observe():
 load=lambda n:json.loads((HERE/n).read_text())if(HERE/n).exists()else{}
 live=load('LIVE_STATUS.json');budget=load('BUDGET.json');jobs=subprocess.check_output(['squeue','-h','-u','zhihan','-o','%i|%j|%T|%C'],text=True).splitlines();active=[line.split('|')for line in jobs if line.split('|')[2]in('RUNNING','CONFIGURING','COMPLETING')]
 hard={tuple([g['type_id'],g['level'],g['controller'],g['state_index']])for g in load('inputs/previous/HARD_CASES.json').get('cases',[])}
 states=[]
 for p in sorted(HERE.glob('SEARCH_*_screen.json')):
  d=json.loads(p.read_text());gs=d['groups'];states.append(dict(code=d['code'],covered16=sum(g['success']and g['first_success']<=16 for g in gs),covered64_lower_bound=sum(g['success']for g in gs),hard_covered=sum(g['success']for g in gs if tuple(g['identity'])in hard),searched=sum(g['searched']for g in gs),pending=(d.get('pending')or{}).get('stage'),complete64=bool(gs)and all(g['success']or g['searched']==64 for g in gs)))
 data=dict(time=datetime.datetime.now().astimezone().isoformat(),stage=live.get('stage'),stage_complete=live.get('stage_complete'),stage_total=live.get('stage_total'),actual_attempts=budget.get('actual_attempts',0),active_cpus=sum(int(r[3])for r in active),cpu_limit=18,protection_event=(HERE/'PROTECTION_VIOLATION.jsonl').exists(),completion=load('COMPLETE.json').get('complete'),searches=states)
 with(HERE/'automatic_observations.jsonl').open('a')as f:f.write(json.dumps(data)+'\n')
 return data

if __name__=='__main__':print(json.dumps(observe(),ensure_ascii=False))
