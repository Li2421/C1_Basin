import sys,os,json,time
os.environ['JAX_PLATFORMS']='cpu'
sys.path.insert(0,'/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness');sys.path.insert(0,'.')
import bootstrap
from common import actions
from design import controllers
from scenes import make_new
from execution import execute
import accel,numpy as np
A=actions();C=controllers();K=[a['key'] for a in A]
S={s['name']:s for s in json.load(open('scenes_design.json'))}
names=sys.argv[1].split(',');same=tot=0;T0=T1=0
for n in names:
 for ci in range(3):
  for a in ('sE22','sE05'):
   c=make_new(S[n],C[ci],S[n]['states'][1],A[K.index(a)],0)
   accel.set_mode(False);t=time.time();r0,a0=execute(c);t0=time.time()-t
   accel.set_mode(True);t=time.time();r1,a1=execute(c);t1=time.time()-t
   eq=all(np.array_equal(a0[k],a1[k]) for k in('positions','velocities','actions','safe','nominal')) and all(r0[k]==r1[k] for k in('success','outcome','episode_length','collision'))
   same+=eq;tot+=1;T0+=t0;T1+=t1
 print(n,same,'/',tot,round(T0,1),round(T1,1),flush=True)
