from common import *
from geometry import make_geometry,task
from catalog import proposal
from design_generated import identities
from registry import register
from dataclasses import asdict

def make_new():
 old=read(D/'multiscene_generalization_v1/CATALOG_FROZEN.json');final=read(D/'multiscene_generalization_v1/CATALOG_FINAL.json')
 used={s['geometry_seed'] for c in [old,final] for ss in c['catalog'].values() for s in ss}
 contents={json.dumps(s['rectangles']) for c in [old,final] for ss in c['catalog'].values() for s in ss}
 result={};reject=[];cases=[];rows=[]
 for f in ['B','F']:
  ordinal='ABCDEFG'.index(f);states=[dict(s,origin='round39 training reused') for s in old['catalog'][f] if s['role']=='train']
  for role,count,base in [('train_new',48,202640000000),('navigation_holdout',8,202650000000)]:
   for j in range(count):
    for k in range(10000):
     gs=base+ordinal*10000000+j*10000+k;ts=202670000000+ordinal*100000000+(0 if role=='train_new' else 50000000)+j*10000+k
     try:
      s=proposal(f,role,j,gs,ts)
      if gs in used or json.dumps(s['rectangles']) in contents:raise ValueError('geometry duplicate')
      s['alias']=f'round40_{f}_{role}_{j}_{gs}_{ts}';s['origin']='round40 new';s['seed_block']=base+ordinal*10000000+j*10000
      used.add(gs);contents.add(json.dumps(s['rectangles']));states.append(s);break
     except ValueError as e:reject.append(dict(family=f,role=role,index=j,seed=gs,reason=str(e)))
    else:raise RuntimeError('seed block exhausted')
    c,r=identities(s,'sequential_expert','expert');cases.append(c);rows+=r
  result[f]=states
 write(HERE/'PILOT_GEOMETRY_CANDIDATES.json',result);write(HERE/'PILOT_GEOMETRY_REJECTIONS.json',reject)
 register('joint_new',cases,rows)

if __name__=='__main__':make_new()
