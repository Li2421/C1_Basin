"""Agent's direct progress check; does not replace active follow-up."""
from pathlib import Path
import os,json,time
os.sched_setaffinity(0,{23});D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
s=read(D/'RESOURCE_STATUS.json');index=D/'CASE_NAMES.json'
if not index.exists():index.write_text(json.dumps({c['rollout_uid']:c['name'] for c in read(D/'ACTIVE_NATIVE_CASES.json')}))
names=read(index);out={k:s[k] for k in ['completed','total','pending','our_capacity']};out['other_cpu']=round(s['other_observed_cpu'],2)
out['active_total']=len(s['active']);out['paused']=sum(v['paused'] for v in s['active'].values());out['running']=[]
for v in [v for v in s['active'].values() if not v['paused']][:3]:
    p=D/'progress'/(v['uid']+'.json');r=read(p) if p.exists() else {}
    out['running'].append(dict(name=names[v['uid']],sim=r.get('time'),horizon=r.get('horizon'),calls=r.get('calls')))
out['failed']=[dict(uid=x['uid'],outcome=x['outcome'],exit=x['exit_code']) for x in read(D/'SUPERVISOR_HISTORY.json') if not x['result_exists'] or x['outcome']=='unknown']
with (D/'logs/AGENT_ACTIVE_CHECKS.jsonl').open('a') as f:f.write(json.dumps(dict(checked_at=time.time(),**out))+'\n')
print(json.dumps(out,ensure_ascii=False))
