from common import *
from catalog import proposal
from design_generated import identities
from registry import register

def finish(stage,attempt):
 catalog=read(HERE/'PILOT_GEOMETRY_CANDIDATES.json');cases=read(HERE/f'{stage}_CASES.json');retry=[];ident=[]
 for c in cases:
  s=c['state'];raw=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");i=next(i for i,q in enumerate(catalog[s['family']]) if q['alias']==s['alias'])
  if raw['success']:
   catalog[s['family']][i].update(joint_feasibility='certified',joint_proof=dict(raw_path=str(HERE/stage/'raw'/f"{c['rollout_uid']}.json"),trace_sha256=raw['trace_sha256'],success=True,outcome=raw['outcome']))
  else:
   for k in range(1,10000):
    gs=s['geometry_seed']+k
    if gs>=s['seed_block']+10000:raise RuntimeError('block exhausted')
    try:q=proposal(s['family'],s['role'],s['index'],gs,s['task_seed']+k);break
    except ValueError:continue
   q.update(origin='round40 new',seed_block=s['seed_block']);q['alias']=q['alias'].replace('round39','round40');catalog[s['family']][i]=q
   nc,r=identities(q,'sequential_expert','expert');retry.append(nc);ident+=r
 write(HERE/'PILOT_GEOMETRY_CANDIDATES.json',catalog)
 if retry:register(f'joint_retry_{attempt}',retry,ident);print('RETRY',attempt,len(retry))
 else:
  for f,ss in catalog.items():
   assert len([s for s in ss if s['role'] in ['train','train_new']])==64
   assert all(s['joint_feasibility']=='certified' for s in ss)
  write(HERE/'PILOT_GEOMETRY_FROZEN.json',catalog);seal('PILOT_GEOMETRY_FROZEN.receipt.json',[HERE/'PILOT_GEOMETRY_FROZEN.json'])
  print('COMPLETE')
if __name__=='__main__':finish(sys.argv[1],int(sys.argv[2]))
