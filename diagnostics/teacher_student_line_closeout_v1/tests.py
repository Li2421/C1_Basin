from core import *
from metrics import *
from ring_exchange.expert import circulation_signature
precision()
def main():
 checks=[];theta=np.linspace(0,1,101);starts=np.arange(4)*np.pi/2
 for signs,expected in [(np.ones(4),'ccw'),(-np.ones(4),'cw'),(np.array([1,-1,1,-1]),'mixed'),(np.zeros(4),'stationary')]:
  angle=starts+theta[:,None]*signs;p=2*np.stack((np.cos(angle),np.sin(angle)),-1);s,m=signature_codes(np.unwrap(np.arctan2(p[...,1],p[...,0]),axis=0)[-1]-np.arctan2(p[0,:,1],p[0,:,0]));assert circulation_signature(p)==expected;assert ['stationary','cw','ccw','mixed'][int(m)]==expected
 checks.append('source circulation_signature agrees on four synthetic trajectory classes')
 s,m=signature_codes(np.array([.15,-.15,0.,0.]));assert not s.any() and m==0;checks.append('strict .15 rad thresholds')
 p0=np.array([[1.8,-.165],[1.8,.165],[-2.,0.],[0.,-2.]]);a=np.array([[0.,.2],[0.,-.2],[0.,0.],[0.,0.]])[None];p=np.stack([p0,p0+DT*a[0]]);s=np.zeros((2,4),int);s[:,0]=1;s[:,1]=-1;c=collision_geometry(p,a,s);assert c['types']==['agent'] and c['head_on_episode'] and c['opposite_head_on_episode'];checks.append('swept robot pair and observed head-on classification')
 p0=np.array([[1.07,0.],[0.,2.],[-2.,0.],[0.,-2.]]);a=np.zeros((1,4,2));a[0,0,0]=-.2;c=collision_geometry(np.stack([p0,p0+DT*a[0]]),a,np.zeros((2,4),int));assert c['types']==['center'];p0[0,0]=3.08;a[0,0,0]=.2;c=collision_geometry(np.stack([p0,p0+DT*a[0]]),a,np.zeros((2,4),int));assert c['types']==['outer'];checks.append('native center and outer collision geometry')
 assert not runs(np.ones((39,4),bool),40) and len(runs(np.ones((40,4),bool),40))==4;checks.append('40-step stall boundary')
 p0=np.array([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]]);p=np.broadcast_to(p0,(701,4,2));m=features(p,np.zeros((700,4,2)),-p0);assert m['stall_runs']==4 and m['stall_robot_seconds']==140 and m['moving_no_progress_robot_seconds']==0 and m['last_progress_step']==0;checks.append('stationary timeout: 140 robot-seconds, no moving stall')
 p=np.broadcast_to(p0,(101,4,2)).copy();p[:,0,1]=np.where(np.arange(101)%2,0.001,0.);a=np.diff(p,axis=0)/DT;m=features(p,a,-p0);assert m['moving_no_progress_robot_seconds']>2;checks.append('motion with negligible target progress detected')
 expert=[]
 for r in read(HERE/'MANIFEST.json'):
  if r['group']!='expert':continue
  with np.load(r['trace_path']) as z:label=circulation_signature(z['positions']);assert label in ['cw','ccw'];expert.append(dict(rollout_uid=r['rollout_uid'],label=label,trace_sha256=r['trace_sha256']))
 assert len(expert)==32
 write(HERE/'UNIT_TESTS.json',dict(utc=utc(),status='PASS',checks=checks,expert_reference=expert,precision=precision(),source_reference=dict(path=str(ROOT/'ring_exchange/expert.py'),sha256=sha(ROOT/'ring_exchange/expert.py')),new_rollouts=0));seal('TESTS.receipt.json',[HERE/'UNIT_TESTS.json',HERE/'tests.py',HERE/'metrics.py'],simulation_steps=0);print('PASS',len(checks),'synthetic checks and32 stored expert traces',flush=True)
if __name__=='__main__':main()
