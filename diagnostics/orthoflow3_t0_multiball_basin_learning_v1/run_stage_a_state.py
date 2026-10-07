#!/usr/bin/env python3
"""Resumable fixed-protocol multiball construction for one frozen t0 state."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,itertools,json,math,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_t0_multiball_basin_learning_v1'
T0=ROOT/'diagnostics/orthoflow3_t0_basin_structure_v1';COMP=ROOT/'diagnostics/orthoflow3_t0_basin_completion_v1'
RUNNER=T0/'run_t0_anchor.py';AFF=np.array([.875,0,.375]);SCALE=np.array([.75,1,.75]);FUTURE=2026092811
RAYS=(.025,.05,.10,.20,.35,.50);KAPPA=.85;ROLLOUT_CHUNK=64

def rows(p): return list(csv.DictReader(open(p)))
def write(p,rr,fields=None):
    fields=fields or (list(rr[0]) if rr else ['state_id'])
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def key(e):return np.asarray(e,np.float64).tobytes().hex()
def nt(e):return (np.asarray(e,float)-AFF)/SCALE
def rt(e):return AFF+SCALE*np.asarray(e,float)
def inside(x,eq):return bool(np.all(eq[:,:3]@x+eq[:,3]<=2e-10))
def clearance(x,eq):return float(np.min(-(eq[:,:3]@x+eq[:,3])/np.linalg.norm(eq[:,:3],axis=1)))
def raylim(x,d,eq):
    v=[float(-(a[:3]@x+a[3])/(a[:3]@d)) for a in eq if a[:3]@d>1e-13]
    return max(0.,min(v)) if v else math.inf
def summary(rr):return sum(bool(x['success']) for x in rr),len(rr)
def tasks(sid,e,n,**kw):return [{'state_id':sid,'eta':np.asarray(e,np.float64).tolist(),'future_index':i,**kw} for i in range(n)]

def load_modules(out,sid):
    spec=importlib.util.spec_from_file_location('t0_runner_runtime',RUNNER);r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
    m=r.load_patched()
    # Operational-only throughput change: preserve every task/RNG/float64 eta tuple,
    # but evaluate 64 rather than 32 independent continuations per JAX call.
    old_source=(m.OLD/'pilot6.py').read_text()
    marker='for begin in range(0,len(tasks),32):'
    slice_marker='chunk=tasks[begin:begin+32]'
    if marker not in old_source or slice_marker not in old_source: raise RuntimeError('rollout chunk patch anchor missing')
    old_source=old_source.replace(marker,f'for begin in range(0,len(tasks),{ROLLOUT_CHUNK}):')
    old_source=old_source.replace(slice_marker,f'chunk=tasks[begin:begin+{ROLLOUT_CHUNK}]')
    ospec=importlib.util.spec_from_loader('multiball_old_runtime',loader=None);old=importlib.util.module_from_spec(ospec);old.__file__=str(m.OLD/'pilot6.py');sys.modules[ospec.name]=old
    exec(compile(old_source,old.__file__,'exec'),old.__dict__)
    old.HERE=out;old.FUTURE_ROOT=FUTURE;old.CAP_CONT=100000;old.CAP_STEPS=50000000
    class Oracle(old.Oracle):
      def _load_prior(self_inner):
        for p,src in [(T0/'anchor_runs'/sid/'raw/pilot_rollouts.jsonl','prior_t0_ball'),(COMP/'raw'/sid/'q64_rollouts.jsonl','prior_q64')]:
          if p.exists():
           for line in p.read_text().splitlines():
            if line.strip():self_inner._insert(json.loads(line),src)
        if self_inner.record_path.exists():
          for line in self_inner.record_path.read_text().splitlines():
           if line.strip():
            x=json.loads(line);self_inner.rows.append(x);self_inner._insert(x,'pilot')
          self_inner.new=len(self_inner.rows);self_inner.steps=sum(int(x['continuation_steps']) for x in self_inner.rows)
    return old,Oracle

def unit_ball_points(skip,n):
    u=qmc.Sobol(3,scramble=False).random_base2(10)[skip:skip+n];out=[]
    for x in u:
      z=1-2*x[0];th=2*math.pi*x[1];a=math.sqrt(max(0,1-z*z));rad=x[2]**(1/3)
      out.append(rad*np.array([a*math.cos(th),a*math.sin(th),z]))
    return out

def build_ball(sid,component,c,center_success,oracle,directions,eq,old,out):
    ct=nt(c);detail=[];rayrows=[];bis=[];prom=[]
    for dr in directions:
      did=dr['direction_id'];d=np.array([float(dr['d1']),float(dr['d2']),float(dr['d3'])]);rd=raylim(ct,d,eq)
      rr=[x for x in RAYS if x<rd-1e-12];near=.95*rd
      if rd>1e-12 and all(abs(near-x)>1e-10 for x in rr):rr.append(near)
      z={'rho':0.,'eta':c,'screen':True,'successes':center_success};detail.append({'direction_id':did,'d':d,'rd':rd,'grid':sorted(set(rr)),'next':0,'tested':[z],'last':z,'fail':None})
    while True:
      ent=[];tt=[]
      for x in detail:
       if x['fail'] is None and x['next']<len(x['grid']):
        rho=x['grid'][x['next']];e=rt(ct+rho*x['d']);ent.append((x,rho,e));tt+=tasks(sid,e,8,component=component,direction_id=x['direction_id'],rho=rho)
      if not ent:break
      got=oracle.ensure(tt,f'C{component}_ray')
      for j,(x,rho,e) in enumerate(ent):
       rr=got[8*j:8*j+8];s,n=summary(rr);y={'rho':rho,'eta':e,'screen':s==8,'successes':s};x['tested'].append(y);x['next']+=1
       rayrows.append({'state_id':sid,'component':component,'direction_id':x['direction_id'],'rho':rho,'eta1':e[0],'eta2':e[1],'eta3':e[2],'successes':s,'trials':n,'screen_8of8':s==8,'stage':'grid'})
       if s==8:x['last']=y
       else:x['fail']=y
    failing=[x for x in detail if x['fail'] is not None]
    for x in failing:x['lo']=x['last']['rho'];x['hi']=x['fail']['rho']
    for bi in range(1,4):
      ent=[];tt=[]
      for x in failing:
       rho=(x['lo']+x['hi'])/2;e=rt(ct+rho*x['d']);ent.append((x,rho,e));tt+=tasks(sid,e,8,component=component,direction_id=x['direction_id'],rho=rho,bisection_round=bi)
      got=oracle.ensure(tt,f'C{component}_bis{bi}') if tt else []
      for j,(x,rho,e) in enumerate(ent):
       rr=got[8*j:8*j+8];s,n=summary(rr);y={'rho':rho,'eta':e,'screen':s==8,'successes':s};x['tested'].append(y)
       bis.append({'state_id':sid,'component':component,'direction_id':x['direction_id'],'round':bi,'rho':rho,'eta1':e[0],'eta2':e[1],'eta3':e[2],'successes':s,'trials':n,'screen_8of8':s==8})
       if s==8:x['lo']=rho;x['last']=y
       else:x['hi']=rho;x['fail']=y
    for x in detail:x['estimate']=x['last']['rho']
    selected=sorted(detail,key=lambda x:(x['estimate'],x['direction_id']))[:8];confirmed=[]
    for rank,x in enumerate(selected):
      good=sorted([y for y in x['tested'] if y['screen']],key=lambda y:y['rho'],reverse=True);pick=None
      for fi,y in enumerate(good):
       rr=oracle.ensure(tasks(sid,y['eta'],64,component=component,direction_id=x['direction_id'],rho=y['rho'],limiting_rank=rank),f'C{component}_limit')
       s,n=summary(rr);ok=s>=63;prom.append({'state_id':sid,'component':component,'direction_id':x['direction_id'],'rank':rank,'fallback':fi,'rho':y['rho'],'successes':s,'trials':n,'B63':ok,'selected':ok})
       if ok:pick=y;break
      if pick is not None:confirmed.append(pick['rho'])
    if len(confirmed)!=8:return None,rayrows,bis,prom,[],'LIMITING_DIRECTION_UNRESOLVED'
    rs=min(confirmed);rd=clearance(ct,eq);rb=KAPPA*min(rs,rd)
    if rb<=0:return None,rayrows,bis,prom,[],'NONPOSITIVE_RADIUS'
    slack=-(eq[:,:3]@ct+eq[:,3])-rb*np.linalg.norm(eq[:,:3],axis=1)
    if np.min(slack)<-1e-10:return None,rayrows,bis,prom,[],'DOMAIN_CONTAINMENT_ERROR'
    # Independent component checks, fixed skip distinct by component.
    unit=unit_ball_points(128+component*32,12);rel=np.array([np.linalg.norm(x) for x in unit]);chosen=[]
    for target in (.25,.60,.90,.97):
      avail=[i for i in range(12) if i not in chosen];chosen.append(min(avail,key=lambda i:(abs(rel[i]-target),i)))
    val=[]
    for i,u in enumerate(unit):
      e=rt(ct+rb*u);rr=oracle.ensure(tasks(sid,e,8,component=component,inside_id=f'C{component}I{i:02d}'),f'C{component}_inside_screen');s,_=summary(rr)
      promote=i in chosen or s<8;fs=s;fn=8
      if promote:rr=oracle.ensure(tasks(sid,e,64,component=component,inside_id=f'C{component}I{i:02d}'),f'C{component}_inside64');fs,fn=summary(rr)
      val.append({'state_id':sid,'component':component,'point_id':f'C{component}I{i:02d}','eta1':e[0],'eta2':e[1],'eta3':e[2],
        'relative_radius':rel[i],'screen_successes':s,'mandatory':i in chosen,'promoted':promote,'successes':fs,'trials':fn,'B63':fs>=63,'false_inclusion':promote and fs<63})
    if any(x['false_inclusion'] for x in val):return None,rayrows,bis,prom,val,'INTERNAL_FALSE_INCLUSION'
    return {'state_id':sid,'component':component,'c1':c[0],'c2':c[1],'c3':c[2],'center_successes':center_success,'center_B63':True,
      'r_success':rs,'r_domain':rd,'r_raw':min(rs,rd),'r_ball':rb,'kappa':KAPPA,'contained':True,'status':'ACCEPTED'},rayrows,bis,prom,val,'ACCEPTED'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
    manifest=json.load(open(HERE/'fixed8_manifest.json'));st=manifest['states'][a.index];sid=st['state_id'];out=HERE/'stage_a'/sid;out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True)
    old,Oracle=load_modules(out,sid);states={sid:st};features=np.load(T0/'synthetic_qdir/conditioning_features.npz')['features'];oracle=Oracle(states,features,np.zeros(3),np.ones(3))
    geom=json.load(open(HERE/'geometry_constants.json'));eq=np.array(geom['halfspaces']);dirs=rows(HERE/'frozen_ray_directions.csv');cloud=rows(HERE/'outward_sobol_cloud.csv')
    original=rows(COMP/'completed_t0_balls.csv');ob=next(x for x in original if x['state_id']==sid)
    balls=[{'state_id':sid,'component':0,'c1':float(ob['c1']),'c2':float(ob['c2']),'c3':float(ob['c3']),'center_successes':64,'center_B63':True,
      'r_success':float(ob['r_success']),'r_domain':float(ob['r_domain']),'r_raw':float(ob['r_raw']),'r_ball':float(ob['r_ball']),'kappa':.85,'contained':True,'status':'ACCEPTED_ORIGINAL'}]
    started=time.time();screen=[]
    construction=[x for x in cloud if x['purpose']=='construction']
    tt=[]
    for x in construction:tt+=tasks(sid,[float(x['eta1']),float(x['eta2']),float(x['eta3'])],8,candidate_index=int(x['global_index']))
    got=oracle.ensure(tt,'outward_screen')
    for i,x in enumerate(construction):
      rr=got[8*i:8*i+8];s,n=summary(rr);screen.append({'state_id':sid,**x,'successes':s,'trials':n,'screen_8of8':s==8,
        'deadlock':sum(y['outcome']=='safe_deadlock' for y in rr),'timeout':sum(y['outcome']=='timeout' for y in rr),'collision':sum(y['outcome']=='collision' for y in rr)})
    write(out/'outward_screening.csv',screen)
    promotions=[];allrays=[];allbis=[];alllim=[];allinside=[];attempted=set()
    while len(balls)<4:
      ranked=[]
      for x in screen:
       idx=int(x['global_index'])
       if not x['screen_8of8'] or idx in attempted:continue
       te=np.array([float(x['t1']),float(x['t2']),float(x['t3'])]);du=min(max(0.,np.linalg.norm(te-nt([b['c1'],b['c2'],b['c3']]))-float(b['r_ball'])) for b in balls)
       if du>=.10-1e-12:ranked.append((-du,idx,x,du))
      ranked.sort(key=lambda z:(z[0],z[1]))
      if not ranked:break
      _,idx,x,du=ranked[0];attempted.add(idx);e=np.array([float(x['eta1']),float(x['eta2']),float(x['eta3'])])
      rr=oracle.ensure(tasks(sid,e,64,candidate_index=idx,outer_center=True),'outer_center64');s,n=summary(rr);b63=s>=63
      rec={'state_id':sid,'candidate_index':idx,'eta1':e[0],'eta2':e[1],'eta3':e[2],'d_union':du,'successes':s,'trials':n,'B63':b63,'component_attempted':'','component_status':'CENTER_NON_B63' if not b63 else ''}
      if b63:
       comp=len(balls);b,ray,bis,lim,val,status=build_ball(sid,comp,e,s,oracle,dirs,eq,old,out);rec['component_attempted']=comp;rec['component_status']=status
       allrays+=ray;allbis+=bis;alllim+=lim;allinside+=val
       if b is not None:balls.append(b)
      promotions.append(rec)
      write(out/'outer_center_promotions.csv',promotions);write(out/'component_balls.csv',balls);write(out/'component_ray_screening.csv',allrays);write(out/'component_bisection.csv',allbis);write(out/'component_limiting_promotions.csv',alllim);write(out/'component_inside_validation.csv',allinside)
    # Freeze and validate the final union.  16 deterministic points cover every component.
    union=[];K=len(balls);udir=unit_ball_points(512,64);pidx=0
    for i in range(16):
      k=i%K;b=balls[k];u=udir[pidx];pidx+=1
      # Force two independent near-boundary points/component among its first two allocations.
      ordinal=i//K; target=.90 if ordinal==0 else (.97 if ordinal==1 else np.linalg.norm(u));u=u/max(np.linalg.norm(u),1e-12)*target
      e=rt(nt([b['c1'],b['c2'],b['c3']])+float(b['r_ball'])*u);rr=oracle.ensure(tasks(sid,e,8,union_id=f'U{i:02d}'),'union_screen');s,n=summary(rr)
      mandatory=ordinal<2;promote=mandatory or s<8;fs=s;fn=n
      if promote:rr=oracle.ensure(tasks(sid,e,64,union_id=f'U{i:02d}'),'union64');fs,fn=summary(rr)
      union.append({'state_id':sid,'point_id':f'U{i:02d}','source_component':k,'relative_radius':target,'eta1':e[0],'eta2':e[1],'eta3':e[2],
        'screen_successes':s,'mandatory_near_boundary':mandatory,'promoted':promote,'successes':fs,'trials':fn,'B63':fs>=63,'false_inclusion':promote and fs<63})
    write(out/'union_inside_validation.csv',union)
    # Independent coverage: screen all; promote every 8/8 point to 64.
    cov=[];coverage=[x for x in cloud if x['purpose']=='independent_coverage'];tt=[]
    for x in coverage:tt+=tasks(sid,[float(x['eta1']),float(x['eta2']),float(x['eta3'])],8,coverage_index=int(x['global_index']))
    got=oracle.ensure(tt,'coverage_screen')
    for i,x in enumerate(coverage):
      e=np.array([float(x['eta1']),float(x['eta2']),float(x['eta3'])]);rr=got[8*i:8*i+8];s,n=summary(rr);promote=s==8;fs=s;fn=n
      if promote:rr=oracle.ensure(tasks(sid,e,64,coverage_index=int(x['global_index'])),'coverage64');fs,fn=summary(rr)
      te=nt(e);inside_union=any(np.linalg.norm(te-nt([b['c1'],b['c2'],b['c3']]))<=float(b['r_ball'])+1e-12 for b in balls)
      cov.append({'state_id':sid,**x,'screen_successes':s,'screen_8of8':s==8,'promoted':promote,'successes':fs,'trials':fn,
        'B63_discovered':promote and fs>=63,'inside_union':inside_union})
    write(out/'independent_coverage_audit.csv',cov)
    robust=[x for x in cov if x['B63_discovered']];coverage_fraction=sum(x['inside_union'] for x in robust)/len(robust) if robust else None
    centers=[nt([b['c1'],b['c2'],b['c3']]) for b in balls];radii=[float(b['r_ball']) for b in balls]
    diameter=max(np.linalg.norm(centers[i]-centers[j])+radii[i]+radii[j] for i in range(K) for j in range(K))
    # Common 2^18 QMC union-volume estimator.
    eng=qmc.Sobol(3,scramble=False);low=np.array([0.,-.5,0.]);high=np.array([1.25,.5,.75]);accepted=[]
    while len(accepted)<2**18:
      for u in eng.random(65536):
       e=low+(high-low)*u;t=nt(e)
       if inside(t,eq):accepted.append(t)
       if len(accepted)==2**18:break
    inside_count=sum(any(np.linalg.norm(t-c)<=r for c,r in zip(centers,radii)) for t in accepted);volume=float(geom['normalized_volume'])*inside_count/len(accepted)
    usable=all([not any(x['false_inclusion'] for x in allinside+union),max(radii)>=.15,diameter>=.30,(K>=2 or max(radii)>=.20)])
    result={'state_id':sid,'K':K,'largest_radius':max(radii),'median_radius':float(np.median(radii)),'union_diameter':float(diameter),
      'union_volume_estimate':volume,'union_volume_fraction':volume/float(geom['normalized_volume']),'union_false_inclusions':sum(x['false_inclusion'] for x in allinside+union),
      'coverage_B63_discovered':len(robust),'coverage_inside_union':sum(x['inside_union'] for x in robust),'robust_coverage':coverage_fraction,
      'training_usable':usable,'new_continuations':oracle.new,'new_physical_steps':oracle.steps,'wall_seconds':time.time()-started,
      'outcomes':dict(Counter(x['outcome'] for x in oracle.rows))}
    dump(out/'state_summary.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
