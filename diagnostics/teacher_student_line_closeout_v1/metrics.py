"""Pure functions over stored trajectories; no environment/controller execution."""
from core import *
DT=.05;TOL=.09;PAIRS=np.array([(i,j) for i in range(4) for j in range(i+1,4)])
precision()
def signature_codes(displacement):
 sign=np.where(displacement>.15,1,np.where(displacement<-.15,-1,0));pos=(sign>0).any(-1);neg=(sign<0).any(-1)
 mode=np.where(pos&neg,3,np.where(pos,2,np.where(neg,1,0))) # 0 stationary,1 CW,2 CCW,3 mixed
 return sign,mode
def runs(mask,min_length=1):
 mask=np.asarray(mask,bool)
 if mask.ndim==1:mask=mask[:,None]
 out=[]
 for i in range(mask.shape[1]):
  change=np.diff(np.r_[False,mask[:,i],False].astype(int));starts=np.flatnonzero(change==1);ends=np.flatnonzero(change==-1)
  out.extend(dict(agent=i,start=int(s+1),end=int(e),steps=int(e-s),seconds=float((e-s)*DT)) for s,e in zip(starts,ends) if e-s>=min_length)
 return out
def segment_closest(a,b):
 d=b-a;alpha=np.clip(-(a*d).sum(-1)/np.maximum((d*d).sum(-1),1e-30),0,1);point=a+alpha[...,None]*d;return np.linalg.norm(point,axis=-1),alpha

def facing(p,v,i,j):
 delta=p[j]-p[i];distance=np.linalg.norm(delta);speed=np.linalg.norm(v[[i,j]],axis=1)
 if distance<1e-12 or np.any(speed<.01):return False
 n=delta/distance;u=v[i]/speed[0];w=v[j]/speed[1]
 return bool(u@n>=.5 and w@(-n)>=.5 and u@w<=-.5)
def collision_geometry(p,a,sign):
 before=p[:-1];after=p[1:];rmin,alph=segment_closest(before,after);obs=rmin-.90-.16<=.005;radii=np.linalg.norm(p,axis=-1);outer=np.minimum(3.25-.16-radii[:-1],3.25-.16-radii[1:])<=.005
 ii,jj=PAIRS.T;rel0=before[:,ii]-before[:,jj];rel1=after[:,ii]-after[:,jj];d,pa=segment_closest(rel0,rel1);pair=d-.32<=0.;events=[]
 anyhit=obs.any(1)|outer.any(1)|pair.any(1)
 if not anyhit.any():return dict(types=[],events=[],step=None,location=None,robot_positions={},geometry_verified=False,head_on_episode=False,opposite_head_on_episode=False)
 t=int(np.flatnonzero(anyhit)[0]);types=[];robots={}
 if pair[t].any():
  types.append('agent')
  for k in np.flatnonzero(pair[t]):
   i,j=map(int,PAIRS[k]);alpha=float(pa[t,k]);q=before[t]+alpha*(after[t]-before[t]);head=facing(before[t],a[t],i,j);opp=bool(sign[t+1,i]*sign[t+1,j]==-1);events.append(dict(type='agent',pair=[i,j],step=t+1,alpha=alpha,location=((q[i]+q[j])/2).tolist(),head_on=head,opposite_circulation=opp));robots[str(i)]=q[i].tolist();robots[str(j)]=q[j].tolist()
 for kind,mask in [('center',obs),('outer',outer)]:
  if mask[t].any():
   types.append(kind)
   for i in np.flatnonzero(mask[t]):
    alpha=float(alph[t,i]) if kind=='center' else float(radii[t+1,i]>=radii[t,i]);q=before[t,i]+alpha*(after[t,i]-before[t,i]);events.append(dict(type=kind,agent=int(i),step=t+1,alpha=alpha,location=q.tolist()));robots[str(i)]=q.tolist()
 return dict(types=types,events=events,step=t+1,location=events[0]['location'],robot_positions=robots,geometry_verified=True,head_on_episode=any(e.get('head_on',False) for e in events),opposite_head_on_episode=any(e.get('head_on',False) and e.get('opposite_circulation',False) for e in events))
def features(p,a,g,v=None):
 p=np.asarray(p,np.float64);a=np.asarray(a,np.float64).reshape(-1,4,2);g=np.asarray(g,np.float64);T=len(a);assert p.shape==(T+1,4,2) and g.shape==(4,2)
 if v is None:v=np.concatenate((np.zeros((1,4,2)),a),axis=0)
 else:v=np.asarray(v,np.float64);assert v.shape==p.shape
 assert np.isfinite(p).all() and np.isfinite(a).all() and np.isfinite(g).all()
 angles=np.unwrap(np.arctan2(p[...,1],p[...,0]),axis=0);dis=angles-angles[0];sign,mode=signature_codes(dis);dist=np.linalg.norm(g-p,axis=-1);total=dist.sum(-1);speed=np.linalg.norm(v,axis=-1);unfinished=dist[-1]>TOL;slow=(speed[1:]<.01)&(dist[1:]>TOL);stalls=runs(slow,40)
 move=np.zeros((T,4),bool)
 if T>=40:move[39:]=(speed[40:]>=.01)&(dist[40:]>TOL)&(dist[:-40]-dist[40:]<.05)
 moveruns=runs(move);team=np.zeros(T,bool)
 if T>=40:team[39:]=(speed[40:].mean(-1)>=.01)&(total[:-40]-total[40:]<.05)&unfinished.any()
 last=0;anchor=total[0]
 for t in range(1,T+1):
  if anchor-total[t]>=.05:last=t;anchor=total[t]
 literal=np.flatnonzero(total[:-1]-total[1:]>=.05);literal_last=int(literal[-1]+1) if len(literal) else 0
 stable=None
 if mode[-1] in [1,2]:
  mismatched=np.flatnonzero(mode!=mode[-1]);stable=int(mismatched[-1]+1) if len(mismatched) else 0
 tangent=np.stack((-p[-1,:,1],p[-1,:,0]),axis=-1)/np.maximum(np.linalg.norm(p[-1],axis=-1,keepdims=True),1e-12)*sign[-1,:,None];geom=0;moving=0
 for i,j in PAIRS:
  if not (unfinished[i] and unfinished[j] and sign[-1,i]*sign[-1,j]==-1):continue
  delta=p[-1,j]-p[-1,i];n=delta/max(np.linalg.norm(delta),1e-12)
  geom+=bool(tangent[i]@n>=.5 and tangent[j]@(-n)>=.5);moving+=facing(p[-1],v[-1],i,j)
 arrivals=[int(np.flatnonzero(dist[:,i]<=TOL)[0]) if np.any(dist[:,i]<=TOL) else None for i in range(4)]
 ix=np.minimum(np.arange(0,701,10),T);early={str(t):dict(mode=int(mode[min(t,T)]),circulating=int((sign[min(t,T)]!=0).sum()),four_same=bool(np.all(sign[min(t,T)]==1) or np.all(sign[min(t,T)]==-1)),early_terminated=T<t,progress=float(total[0]-total[min(t,T)])) for t in [100,200,300]}
 return dict(T=T,total_time=T*DT,mode=int(mode[-1]),direction=sign[-1].tolist(),angular_displacement=dis[-1].tolist(),circulating=int((sign[-1]!=0).sum()),four_same=bool(np.all(sign[-1]==1) or np.all(sign[-1]==-1)),early=early,stable_step=stable,first_goal_entry_step=arrivals,final_unfinished=int(unfinished.sum()),unfinished_mask=unfinished.tolist(),last_progress_step=last,last_progress_seconds=last*DT,literal_last_progress_step=literal_last,geometric_opposite_pairs=int(geom),moving_opposite_pairs=int(moving),stall_runs=len(stalls),stall_robot_seconds=float(sum(s['seconds'] for s in stalls)),stall_max_seconds=max([s['seconds'] for s in stalls],default=0),stall_first_step=min([s['start'] for s in stalls],default=None),stall_segments=stalls,moving_no_progress_robot_seconds=float(move.sum()*DT),moving_no_progress_max_seconds=max([s['seconds'] for s in moveruns],default=0),moving_no_progress_team_seconds=float(team.sum()*DT),final_positions=p[-1].tolist(),last40_mean_positions=p[-min(40,T+1):].mean(0).tolist(),goals=g.tolist(),final_radii=np.linalg.norm(p[-1],axis=-1).tolist(),distance_curve=total[ix].tolist(),mode_curve=mode[ix].tolist(),steps_curve=np.arange(0,701,10).tolist(),pre_switch_progress_rate=float((total[400]-total[500])/(100*DT)) if T>=600 else None,post_switch_progress_rate=float((total[500]-total[600])/(100*DT)) if T>=600 else None,absorbed_pre_switch_rate=float((total[min(400,T)]-total[min(500,T)])/(100*DT)),absorbed_post_switch_rate=float((total[min(500,T)]-total[min(600,T)])/(100*DT)),kinematic_max_abs_error=float(np.max(np.abs(np.diff(p,axis=0)-DT*a))) if T else 0.)
