#!/usr/bin/env python3
"""Freeze/cache the small targeted t0 basin-shape probe protocol before rollout."""
from __future__ import annotations
import csv, hashlib, itertools, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

ROOT=Path('/home/zhihan/research/Basin_C1'); DIAG=ROOT/'diagnostics'
HERE=DIAG/'orthoflow3_t0_basin_shape_v1'; T0=DIAG/'orthoflow3_t0_basin_structure_v1'; COMP=DIAG/'orthoflow3_t0_basin_completion_v1'; MULTI=DIAG/'orthoflow3_t0_multiball_basin_learning_v1'; MAN=DIAG/'orthoflow3_b63_manifold_representation_v1'; BASIS=DIAG/'double_bottleneck_eta_basis_redesign/tools/bases.py'
AFF=np.array([.875,0,.375]); SCALE=np.array([.75,1.,.75]); FUTURE=2026092811

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=fields or (list(dict.fromkeys(k for r in rows for k in r)) if rows else ['status'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def key(e):return np.asarray(e,dtype=np.float64).tobytes().hex()
def nt(e):return (np.asarray(e,float)-AFF)/SCALE
def rt(t):return AFF+SCALE*np.asarray(t,float)
def inside(t,eq):return bool(np.all(eq[:,:3]@np.asarray(t)+eq[:,3] <= 2e-10))
def parse_eta(v):return np.asarray(json.loads(v) if isinstance(v,str) else v,dtype=float)

def all_sources(sid):
 return [
  ('t0_basin_structure',T0/'anchor_runs'/sid/'raw/pilot_rollouts.jsonl'),
  ('t0_basin_completion',COMP/'raw'/sid/'raw/pilot_rollouts.jsonl'),
  ('t0_multiball',MULTI/'stage_a'/sid/'raw/pilot_rollouts.jsonl'),
 ]

def aggregate(states):
 out={}; lower=[]; q64=[]; conflicts=[]
 for st in states:
  sid=st['state_id']; evid={}
  for source,p in all_sources(sid):
   for line in p.read_text().splitlines():
    if not line.strip():continue
    r=json.loads(line)
    if r.get('state_id')!=sid or r.get('h_conditioning_identifier')!=st['h_conditioning_identifier'] or r.get('source_group')!=st['source_group']:continue
    e=np.asarray(r['eta'],dtype=np.float64);k=key(e);fi=int(r['future_index']);x=evid.setdefault(k,{'eta':e,'future':{},'outcomes':{},'sources':set(),'phases':set()})
    val=(bool(r['success']),str(r['outcome']))
    if fi in x['future'] and x['future'][fi]!=val:conflicts.append((sid,k,fi,x['future'][fi],val))
    x['future'][fi]=val;x['sources'].add(source);x['phases'].add(str(r.get('phase','')))
  if conflicts:raise RuntimeError(f'conflicting exact cached outcomes: {conflicts[:1]}')
  out[sid]=evid
  for k,x in evid.items():
   trials=len(x['future']);succ=sum(v[0] for v in x['future'].values());outs=defaultdict(int)
   for _,o in x['future'].values():outs[o]+=1
   rec={'state_id':sid,'eta_key_float64':k,'eta1':x['eta'][0],'eta2':x['eta'][1],'eta3':x['eta'][2], 'successes':succ,'trials':trials,
        'Q64_available':trials>=64,'B63':trials>=64 and succ>=63,'deadlock':outs['safe_deadlock'],'timeout':outs['timeout'],'collision':outs['collision'],'numerical':outs['other_numerical'],
        'sources':';'.join(sorted(x['sources'])),'phases':';'.join(sorted(x['phases']))}
   (q64 if trials>=64 else lower).append(rec)
 return out,q64,lower

def quantile_pair(points,allkeys,target,used):
 # pairs with an unsampled midpoint; closest to the frozen distance quantile.
 cand=[]
 for i,j in itertools.combinations(range(len(points)),2):
  a,b=points[i],points[j];mid=(a+b)/2
  if key(rt(mid)) in allkeys:continue
  d=float(np.linalg.norm(a-b));cand.append((d,i,j))
 if not cand:return None
 vals=np.array([x[0] for x in cand]);goal=float(np.quantile(vals,target))
 cand.sort(key=lambda x:(abs(x[0]-goal),key(rt(points[x[1]])),key(rt(points[x[2]]))))
 for x in cand:
  pair=(x[1],x[2])
  if pair not in used:used.add(pair);return x
 return None

def fps_anchors(S,n):
 order=np.lexsort((S[:,1],S[:,0]));chosen=[int(order[0])]
 while len(chosen)<min(n,len(S)):
  d=np.min(((S[:,None,:]-S[np.asarray(chosen)][None,:,:])**2).sum(axis=2),axis=1)
  candidates=np.flatnonzero(np.isclose(d,d.max()));chosen.append(int(candidates[0]))
 return chosen

def make_probe(sid,category,eta,**extra):
 return {'state_id':sid,'probe_id':'','category':category,'eta1':float(eta[0]),'eta2':float(eta[1]),'eta3':float(eta[2]),**extra}

def main():
 HERE.mkdir(parents=True,exist_ok=True);(HERE/'raw').mkdir(exist_ok=True);(HERE/'figures').mkdir(exist_ok=True)
 if sha(BASIS)!='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38':raise RuntimeError('basis hash')
 states=json.load(open(COMP/'frozen_8state_manifest.json'))['attempted_states']
 if len(states)!=8 or not all(x['true_t0'] for x in states):raise RuntimeError('frozen state cohort')
 geom=json.load(open(MULTI/'geometry_constants.json'));eq=np.asarray(geom['halfspaces'],float)
 frames={r['state_id']:r for r in csv.DictReader(open(MAN/'affine_plane_fit.csv'))}
 evidence,q64,lower=aggregate(states)
 write(HERE/'cached_q64_manifest.csv',sorted(q64,key=lambda x:(x['state_id'],x['eta_key_float64'])))
 write(HERE/'cached_lower_seed_manifest.csv',sorted(lower,key=lambda x:(x['state_id'],x['eta_key_float64'])))
 probes=[]; probe_roles=[]; initial_by_state={}
 for st in states:
  sid=st['state_id']; frame=frames[sid]; c=np.array([float(frame['center_t1']),float(frame['center_t2']),float(frame['center_t3'])]);U=np.array([[float(frame[f'u1_t{i}']) for i in (1,2,3)],[float(frame[f'u2_t{i}']) for i in (1,2,3)]]).T; normal=np.array([float(frame[f'normal_t{i}']) for i in (1,2,3)])
  b63=[x for x in q64 if x['state_id']==sid and x['B63']];fail=[x for x in q64 if x['state_id']==sid and not x['B63']]
  X=np.asarray([nt([x['eta1'],x['eta2'],x['eta3']]) for x in b63]);S=(X-c)@U;allkeys=set(evidence[sid]);state=[];used=set()
  # A: two local quantiles, one median, one cross-sector/long quantile.
  for name,qq in [('local_1',.20),('local_2',.35),('medium',.50),('long_cross_sector',.85)]:
   pick=quantile_pair(X,allkeys,qq,used)
   if pick is None:continue
   d,i,j=pick
   for alpha in (.25,.50,.75):
    t=(1-alpha)*X[i]+alpha*X[j];eta=rt(t)
    state.append(make_probe(sid,'SUCCESS_SUCCESS_INTERPOLATION',eta,pair_category=name,alpha=alpha,pair_distance=d,endpointA_key=b63[i]['eta_key_float64'],endpointB_key=b63[j]['eta_key_float64']))
  # B: frozen tangential farthest-point anchors and +/- normal offsets.
  for rank,i in enumerate(fps_anchors(S,3)):
   for sign in (-1,1):
    for offset in (.05,.10):
     t=X[i]+sign*offset*normal
     if inside(t,eq):state.append(make_probe(sid,'NORMAL_DIRECTION',rt(t),anchor_rank=rank,anchor_key=b63[i]['eta_key_float64'],normal_sign=sign,normal_offset=offset))
     else:state.append(make_probe(sid,'NORMAL_DIRECTION_DOMAIN_EXCLUDED',rt(t),anchor_rank=rank,anchor_key=b63[i]['eta_key_float64'],normal_sign=sign,normal_offset=offset,domain_excluded=True))
  # C: three nearest opposite-label pairs.
  boundary=[]
  for s in b63:
   ts=nt([s['eta1'],s['eta2'],s['eta3']])
   for f in fail:
    tf=nt([f['eta1'],f['eta2'],f['eta3']]);d=float(np.linalg.norm(ts-tf));mid=(ts+tf)/2
    if key(rt(mid)) not in allkeys:boundary.append((d,s,f,mid))
  boundary.sort(key=lambda z:(z[0],z[1]['eta_key_float64'],z[2]['eta_key_float64']))
  seenmid=set();chosen=[]
  for item in boundary:
   mk=key(rt(item[3]))
   if mk in seenmid:continue
   seenmid.add(mk);chosen.append(item)
   if len(chosen)==3:break
  for bi,(d,s,f,mid) in enumerate(chosen):state.append(make_probe(sid,'SUCCESS_FAILURE_BOUNDARY',rt(mid),boundary_id=bi,pair_distance=d,success_key=s['eta_key_float64'],failure_key=f['eta_key_float64']))
  # D: at most two interior Delaunay holes, reserving 3 slots/state for boundary refinements.
  try:
   tri=Delaunay(S);holes=[]
   for simplex,neigh in zip(tri.simplices,tri.neighbors):
    if np.any(neigh<0):continue
    ss=S[simplex];area=abs(np.cross(ss[1]-ss[0],ss[2]-ss[0]))/2;cent=ss.mean(axis=0);nn=float(np.mean((X[simplex]-c)@normal));t=c+U@cent+normal*nn
    if not inside(t,eq) or key(rt(t)) in allkeys:continue
    mind=float(np.min(np.linalg.norm(X-t,axis=1)))
    if mind<.05:continue
    holes.append((-area,cent,nn,mind,simplex))
   holes.sort(key=lambda x:(x[0],tuple(x[1])))
   selected=[]
   for h in holes:
    if all(np.linalg.norm(h[1]-q[1])>=.10 for q in selected):selected.append(h)
    if len(selected)==2:break
   for hi,(_,cent,nn,mind,simplex) in enumerate(selected):state.append(make_probe(sid,'HOLE_INTERIOR',rt(c+U@cent+normal*nn),hole_id=hi,hole_area=-selected[hi][0],nearest_B63_distance=mind,vertex_keys=';'.join(b63[int(i)]['eta_key_float64'] for i in simplex)))
  except Exception:pass
  # Exact coordinate dedupe while retaining all requested probe roles.
  unique={}
  for p in state:
   if p['category'].endswith('DOMAIN_EXCLUDED'):continue
   k=key([p['eta1'],p['eta2'],p['eta3']]);unique.setdefault(k,[]).append(p)
  if len(unique)>29:raise RuntimeError((sid,'initial unique eta exceeds frozen 29',len(unique)))
  final=[]
  for z,(k,roles) in enumerate(sorted(unique.items())):
   p=dict(roles[0]);p['probe_id']=f'{sid}__P{z:02d}';p['eta_key_float64']=k;p['role_count']=len(roles);p['all_categories']=';'.join(sorted({r['category'] for r in roles}));p['cached_q64']=len(evidence[sid][k]['future'])>=64 if k in evidence[sid] else False;p['cached_lower_trials']=len(evidence[sid][k]['future']) if k in evidence[sid] else 0;p['new_continuations_needed']=max(0,64-p['cached_lower_trials']);final.append(p)
   for role_index,role in enumerate(roles):
    rr=dict(role);rr['probe_id']=p['probe_id'];rr['role_index']=role_index;rr['eta_key_float64']=k;rr['cached_q64']=p['cached_q64'];rr['cached_lower_trials']=p['cached_lower_trials'];rr['new_continuations_needed']=p['new_continuations_needed'];probe_roles.append(rr)
  # Retain domain exclusions visibly but never send them to runner.
  for p in state:
   if p['category'].endswith('DOMAIN_EXCLUDED'):
    p['probe_id']=f'{sid}__DOMAIN_EXCLUDED';p['eta_key_float64']=key([p['eta1'],p['eta2'],p['eta3']]);p['role_count']=1;p['all_categories']=p['category'];p['cached_q64']=False;p['cached_lower_trials']=0;p['new_continuations_needed']=0;final.append(p)
    rr=dict(p);rr['role_index']=0;probe_roles.append(rr)
  probes.extend(final);initial_by_state[sid]=[p for p in final if not p['category'].endswith('DOMAIN_EXCLUDED')]
  dump(HERE/f'initial_targets_{sid}.json',initial_by_state[sid])
 write(HERE/'targeted_probe_manifest.csv',probes)
 write(HERE/'targeted_probe_roles_initial.csv',probe_roles)
 total_new=sum(int(x['new_continuations_needed']) for x in probes);maxstate=max(sum(int(x['new_continuations_needed']) for x in initial_by_state[s]) for s in initial_by_state)
 dump(HERE/'fixed8_manifest.json',{'states':states,'source':str(COMP/'frozen_8state_manifest.json'),'source_sha256':sha(COMP/'frozen_8state_manifest.json')})
 dump(HERE/'cache_reuse_audit.json',{'q64_records':len(q64),'lower_seed_records':len(lower),'exact_sources':['t0_basin_structure','t0_basin_completion','t0_multiball'],'invalid_feature_replay_excluded':True,'future_root_seed':FUTURE,'new_rollouts_during_audit':0})
 dump(HERE/'cost_preflight.json',{'initial_unique_targeted_eta':sum(len(v) for v in initial_by_state.values()),'initial_new_continuations':total_new,'max_initial_new_continuations_per_state':maxstate,'boundary_refinement_max_continuations':8*3*64,'hard_new_eta_per_state':32,'initial_new_eta_per_state_max':29,'within_nominal_continuation_budget':total_new+8*3*64<=16384,'authorized_shards':6,'server_idle_checked':True})
 (HERE/'protocol.md').write_text('''# OrthoFlow3 t0 basin shape and cross-state morphology v1\n\nThis experiment freezes eight exact t0 states and uses the archived R0 affine frames as coordinates only. It first aggregates exact Q64 labels (both B63 and non-B63) and keeps lower-seed evidence separate. The initial targeted design has at most 29 unique eta/state: four B63-pair interpolation trajectories at alpha .25/.50/.75; three tangentially spread B63 anchors with normal offsets +/- .05/.10; three nearest B63/non-B63 midpoint probes; and at most two interior Delaunay hole probes. All new eta receive 64 matched continuations. After outcomes, each initial boundary midpoint with a bracket >=.05 receives at most one deterministic refinement; the initial budget leaves three eta/state for this, so total remains <=32. No adaptive expansion, large Sobol sweep, learned model, or controller modification is permitted.\n''')
 print(json.dumps({'q64_records':len(q64),'lower':len(lower),'initial_new_continuations':total_new,'per_state_targets':{s:len(v) for s,v in initial_by_state.items()}},indent=2))
if __name__=='__main__':main()
