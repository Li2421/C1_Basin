"""Candidate sibling scenes for five need clusters (scratch, not evidence).
Each scene has 24 states: 0-7 design (selection), 8-23 fresh (verification / validation)."""
import sys,json,math
import numpy as np
H='/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness'
sys.path.insert(0,H)
import bootstrap  # noqa
from common import uid
from legacy_states import LiteratureConfig
from bottleneck_family.scenario import Config,Geometry
from single_integrator.environment import point_segment_distance
from swap_geometry_user_fix_20261023 import static_walls
SEED=2026102691;NS=24
def ring(name,n,R,perturb,rng,lateral=None):
 th=np.arange(n)*2*np.pi/n;ini=R*np.stack([np.cos(th),np.sin(th)],-1);goals=-ini;out=[]
 for i in range(NS):
  if i==0:p=ini.copy()
  elif lateral is None:p=ini+rng.uniform(-perturb,perturb,ini.shape)
  else:
   radial=ini/R;tan=np.stack([-radial[:,1],radial[:,0]],-1);p=ini+rng.uniform(-perturb,perturb,n)[:,None]*radial+rng.uniform(-lateral,lateral,n)[:,None]*tan
  out.append(dict(scene=name,index=i,N=n,positions=p.tolist(),goals=goals.tolist(),velocities=np.zeros_like(p).tolist()))
 return out
def finish(name,cluster,states,cfg,walls,bc=None,free_area=None,opening=10.,diametric=True,params=None):
 if bc:
  for s in states:s['bottleneck_config']=bc
 sc=dict(name=name,title=name,family='hetset',factor=cluster,cluster=cluster,params=params,config=cfg,walls=walls,states=states,free_area=free_area,
  min_opening_over_diameter=opening,serial_walls=0,diametric=diametric,moving=False)
 if bc:sc['bottleneck_config']=bc
 sc['definition_sha256']=uid('scene_definition',sc);return sc
def longest(states,v):return max(np.linalg.norm(np.asarray(s['goals'])-s['positions'],axis=-1).max() for s in states)/v
def plus(w,tag):
 name=f'xsc_hs_plus_w{round(w*100):03d}{tag}';bc=Config(num_agents=4,half_length=1.6,half_height=1.6,barrier_x=(1.35,),openings=(((0.,2.8),),),barrier_thickness=.1,extra_rectangles=((w,w,1.6,1.6),(-1.6,w,-w,1.6),(-1.6,-1.6,-w,-w),(w,-1.6,1.6,-w)),agent_radius=.05,dt=.05,max_speed=.5,max_steps=400)
 rng=np.random.default_rng(np.random.SeedSequence([SEED,1,round(1000*w)]))
 th=np.arange(4)*2*np.pi/4;radial=np.stack([np.cos(th),np.sin(th)],-1);tan=np.stack([-np.sin(th),np.cos(th)],-1);ini=radial.copy();goals=ini[(np.arange(4)+2)%4];st=[]
 for i in range(NS):
  p=ini.copy() if i==0 else ini+rng.uniform(-.1,.1,4)[:,None]*radial+rng.uniform(-.02,.02,4)[:,None]*tan
  st.append(dict(scene=name,index=i,N=4,positions=p.tolist(),goals=goals.tolist(),velocities=np.zeros_like(p).tolist()))
 cfg=LiteratureConfig(name,4,.05,.05,.5,int(math.ceil(4.*longest(st,.5)/.05))).to_dict()
 return finish(name,'plus',st,cfg,Geometry(bc).walls.tolist(),bc.to_dict(),3.2**2-4*(1.6-w)**2,w/.05,True,dict(w=w))
def box(n,r,f,tag=''):
 name=f'xsc_hs_box_n{n:02d}_r{round(r*100):03d}_f{round(f*10):02d}{tag}';bc=Config(num_agents=n,half_length=1.2,half_height=1.2,barrier_x=(.95,),openings=(((0.,2.),),),barrier_thickness=.1,extra_rectangles=(),agent_radius=.05,dt=.05,max_speed=.5,max_steps=400);geom=Geometry(bc)
 rng=np.random.default_rng(np.random.SeedSequence([SEED,2,n,round(100*r),round(10*f)]));clear=r+.035;st=[]
 ok=lambda p:bool(np.all(point_segment_distance(p,geom.walls[:,0],geom.walls[:,1])>clear))
 while len(st)<NS:
  p=[];g=[]
  for _ in range(2000):
   x=rng.uniform(-1.2,1.2,2)
   if ok(x) and (not p or np.all(np.linalg.norm(np.asarray(p)-x,axis=-1)>2*r+.1)):p.append(x)
   if len(p)==n:break
  if len(p)!=n:continue
  for _ in range(4000):
   x=rng.uniform(-1.2,1.2,2);i=len(g)
   if ok(x) and np.linalg.norm(x-p[i])>=1. and (not g or np.all(np.linalg.norm(np.asarray(g)-x,axis=-1)>2*r+.1)):g.append(x)
   if len(g)==n:break
  if len(g)!=n:continue
  st.append(dict(scene=name,index=len(st),N=n,positions=np.asarray(p).tolist(),goals=np.asarray(g).tolist(),velocities=np.zeros((n,2)).tolist()))
 cfg=LiteratureConfig(name,n,r,.05,.5,int(math.ceil(f*longest(st,.5)/.05))).to_dict()
 return finish(name,'box',st,cfg,geom.walls.tolist(),bc.to_dict(),2.4**2-.04,10.,False,dict(N=n,r=r,f=f))
def obst(n,R,r,tag):
 # physics scene name must stay lit_swap8_obstacles (walls and nominal route are keyed on it)
 name=f'lit_swap8_obstacles'  ;sname=f'hs_obst_n{n:02d}_R{round(R*100):03d}_r{round(r*1000):03d}{tag}'
 rng=np.random.default_rng(np.random.SeedSequence([SEED,3,n,round(100*R),round(100*r)]+([round(1000*r)] if round(1000*r)%10 else [])))
 st=ring(name,n,R,.1,rng)
 for s in st:s['hs_name']=sname
 cfg=LiteratureConfig(name,n,r,.05,.5,int(round((4*2*R/.5)/.05))).to_dict()
 sc=finish(sname,'obst',st,cfg,static_walls('lit_swap8_obstacles').tolist(),None,None,None,True,dict(N=n,R=R,r=r));return sc
def circle(n,r,D,tag):
 name='lit_circle_five';sname=f'hs_circle_n{n}_r{round(r*10):02d}_D{D}{tag}'
 rng=np.random.default_rng(np.random.SeedSequence([SEED,4,n,round(10*r),D]))
 def cset():
  pts=[]
  for _ in range(200000):
   th=rng.uniform(0,2*np.pi);rr=D*np.sqrt(rng.random());p=rr*np.array([np.cos(th),np.sin(th)])
   if np.linalg.norm(p)+r>D or any(np.linalg.norm(p-q)<=2*r for q in pts):continue
   pts.append(p)
   if len(pts)==n:return pts
  raise RuntimeError
 st=[]
 for i in range(NS):
  p=np.asarray(cset());g=np.asarray(cset());st.append(dict(scene=name,index=i,N=n,positions=p.tolist(),goals=g.tolist(),velocities=np.zeros_like(p).tolist(),hs_name=sname))
 ang=(np.arange(128)+.5)*2*np.pi/128;v=D/np.cos(np.pi/128)*np.stack([np.cos(ang),np.sin(ang)],-1);walls=np.stack([v,np.roll(v,-1,0)],1).tolist()
 cfg=LiteratureConfig(name,n,r,.05,0.37365109585434214,2000).to_dict()
 vv=np.asarray(walls)[:,0];area=abs(np.sum(vv[:,0]*np.roll(vv[:,1],-1)-vv[:,1]*np.roll(vv[:,0],-1)))/2
 return finish(sname,'circle',st,cfg,walls,None,area,2*D/(2*r),False,dict(N=n,r=r,D=D))
def swap(n,R,tag=''):
 name=f'xsc_hs_swap_n{n:02d}{tag}';rng=np.random.default_rng(np.random.SeedSequence([SEED,5,n]));st=ring(name,n,R,.1,rng)
 cfg=LiteratureConfig(name,n,.05,.05,.5,int(math.ceil(4.*longest(st,.5)/.05))).to_dict()
 return finish(name,'swap',st,cfg,[],None,None,10.,True,dict(N=n,R=R))
if __name__=='__main__':
 S=[]
 for w in [.18,.22,.28,.35,.45,.50]:S.append(plus(w,''))
 for n,r,f in [(9,.14,2.5),(10,.12,2.5),(10,.14,2.7),(11,.13,2.5),(8,.14,2.4),(9,.13,2.8),(10,.13,3.0),(8,.16,2.6)]:S.append(box(n,r,f))
 for n,R,r in [(8,1.4,.05),(8,1.3,.05),(8,1.5,.05),(8,1.4,.06),(8,1.4,.04),(7,1.4,.05),(9,1.4,.05),(10,1.5,.05)]:S.append(obst(n,R,r,''))
 for n,r,D in [(5,2.0,11),(4,2.0,11),(6,2.0,11),(5,1.7,11),(5,2.3,11),(5,2.0,10),(6,1.8,11),(4,2.4,11)]:S.append(circle(n,r,D,''))
 for n in [5,7,8,9,11]:S.append(swap(n,n/8))
 json.dump(S,open('scenes_design.json','w'));print(len(S),[s['name'] for s in S])
