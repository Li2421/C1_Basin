#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import qmc
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_analytic_basin_margin_learning_v1';POINT=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1';GEOM=ROOT/'diagnostics/orthoflow3_t0_multiball_basin_learning_v1/geometry_constants.json'
G=json.load(open(GEOM));AFF=np.array(G['normalization']['affine']);SCALE=np.array(G['normalization']['scale']);HS=np.array(G['halfspaces']);GT=.85;GN=.75
def norm(v):return (np.asarray(v,float)-AFF)/SCALE
def domain(z):z=np.atleast_2d(z);return np.all(z@HS[:,:3].T+HS[:,3]<=1e-9,axis=1)
def rows(p):return list(csv.DictReader(open(p)))
def write(p,rr,fields=None):
 rr=list(rr);fields=fields or (list(rr[0]) if rr else [])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)

def ellipse(P2):
 mu=np.mean(P2,axis=0);C=np.cov(P2.T,bias=True) if len(P2)>1 else np.eye(2)*.01
 C=np.atleast_2d(C);w,V=np.linalg.eigh(C);mx=max(float(w.max()),.0025);w=np.maximum(w,max(.0025,.05*mx));C=V@np.diag(w)@V.T;Ci=np.linalg.inv(C)
 base=max(math.sqrt(float((x-mu)@Ci@(x-mu))) for x in P2);base=max(base,1.)
 return mu,Ci,base
def quad(s):
 s=np.atleast_2d(s);return np.c_[np.ones(len(s)),s[:,0],s[:,1],s[:,0]**2,s[:,0]*s[:,1],s[:,1]**2]
def fit_one(kind,P,neg,pt=2.,pad=.15,nf=1.,exponents=(4,2),labels=None):
 P=np.asarray(P,float);neg=np.asarray(neg,float).reshape(-1,3)
 if kind=='ncb_affine':
  mu,Ci,base=ellipse(P[:,:2]);X=np.c_[np.ones(len(P)),P[:,:2]];beta=np.linalg.lstsq(X,P[:,2],rcond=None)[0];res=P[:,2]-X@beta
  mid=(float(res.min())+float(res.max()))/2;half=((float(res.max())-float(res.min()))/2+pad)*nf
  par={'kind':kind,'mu':mu.tolist(),'Ci':Ci.tolist(),'base':base,'tan_expand':pt,'beta':beta.tolist(),'lo_res':mid-half,'hi_res':mid+half,'normal_pad':pad,'normal_factor':nf}
 elif kind=='racs':
  c=P.mean(0);C=np.cov((P-c).T,bias=True) if len(P)>1 else np.eye(3);w,V=np.linalg.eigh(C);order=np.argsort(w)[::-1];U=V[:,order[:2]];v=V[:,order[2]]
  s=(P-c)@U;n=(P-c)@v;mu,Ci,base=ellipse(s);F=quad(s);ridge=np.diag([0,1e-3,1e-3,1e-3,1e-3,1e-3]);beta=np.linalg.solve(F.T@F+ridge,F.T@n);res=n-F@beta
  mid=(float(res.min())+float(res.max()))/2;half=((float(res.max())-float(res.min()))/2+pad)*nf
  par={'kind':kind,'c':c.tolist(),'U':U.tolist(),'v':v.tolist(),'mu':mu.tolist(),'Ci':Ci.tolist(),'base':base,'tan_expand':pt,'beta':beta.tolist(),'lo_res':mid-half,'hi_res':mid+half,'normal_pad':pad,'normal_factor':nf}
 elif kind=='rfse':
  c=P.mean(0);C=np.cov((P-c).T,bias=True) if len(P)>1 else np.eye(3);w,V=np.linalg.eigh(C);R=V[:,np.argsort(w)[::-1]];z=(P-c)@R;ax=np.maximum(np.max(np.abs(z),axis=0),.05);ax[:2]*=pt;ax[2]=(ax[2]+pad)*nf
  par={'kind':kind,'c':c.tolist(),'R':R.tolist(),'axes':ax.tolist(),'p_t':int(exponents[0]),'p_n':int(exponents[1]),'tan_expand':pt,'normal_pad':pad,'normal_factor':nf}
 else:raise ValueError(kind)
 return par
def contains(par,Z,ret=False):
 Z=np.atleast_2d(np.asarray(Z,float));kind=par['kind'];inside=domain(Z);gt=GT if ret else 1.;gn=GN if ret else 1.
 if kind=='ncb_affine':
  x=Z[:,:2];mu=np.array(par['mu']);Ci=np.array(par['Ci']);rad=par['base']*par['tan_expand']*gt;q=np.einsum('ni,ij,nj->n',x-mu,Ci,x-mu)<=rad*rad+1e-10
  mid=x@np.array(par['beta'])[1:]+par['beta'][0]+(par['lo_res']+par['hi_res'])/2;half=(par['hi_res']-par['lo_res'])/2*gn;nok=(Z[:,2]>=mid-half-1e-10)&(Z[:,2]<=mid+half+1e-10);inside&=q&nok
 elif kind=='racs':
  c=np.array(par['c']);U=np.array(par['U']);v=np.array(par['v']);s=(Z-c)@U;n=(Z-c)@v;mu=np.array(par['mu']);Ci=np.array(par['Ci']);rad=par['base']*par['tan_expand']*gt;q=np.einsum('ni,ij,nj->n',s-mu,Ci,s-mu)<=rad*rad+1e-10
  mid=quad(s)@np.array(par['beta'])+(par['lo_res']+par['hi_res'])/2;half=(par['hi_res']-par['lo_res'])/2*gn;inside&=q&(n>=mid-half-1e-10)&(n<=mid+half+1e-10)
 elif kind=='rfse':
  c=np.array(par['c']);R=np.array(par['R']);z=np.abs((Z-c)@R);ax=np.array(par['axes'])*np.array([gt,gt,gn]);val=(z[:,0]/ax[0])**par['p_t']+(z[:,1]/ax[1])**par['p_t']+(z[:,2]/ax[2])**par['p_n'];inside&=val<=1+1e-10
 elif kind in ('two_lobe','two_rap'):inside &= np.any(np.stack([contains(q,Z,ret) for q in par['components']],axis=1),axis=1)
 elif kind=='clipped_ncb':
  inside &= contains(par['base'],Z,ret)
  for hp in par['clips']:
   u=np.array(hp['u']);margin=.025 if ret else 0.;inside &= (Z@u-float(hp['threshold'])>=margin-1e-10)
 elif kind=='rap':
  c=np.array(par['c']);R=np.array(par['R']);z=(Z-c)@R;lo=np.array(par['lo']);hi=np.array(par['hi']);mid=(lo+hi)/2;gam=np.array([GT,GT,GN]) if ret else np.ones(3);lor=mid+gam*(lo-mid);hir=mid+gam*(hi-mid);inside &= np.all((z>=lor-1e-10)&(z<=hir+1e-10),axis=1)
 return inside
def fit_state(kind,P,N,exponents=(4,2)):
 # Fixed deterministic constrained grid: maximize an analytic volume proxy while
 # retaining zero cached negatives after the preregistered erosion.
 best=None
 for pt in [2.,1.75,1.5,1.25,1.,.9,.75,.6,.5,.4,.3]:
  for pad in [.15,.10,.075,.05,.025]:
   for nf in [1.,.85,.70,.55,.40]:
    p=fit_one(kind,P,N,pt,pad,nf,exponents);bad=int(contains(p,N,True).sum()) if len(N) else 0
    full=int(contains(p,P,False).sum());retfit=int(contains(p,P,True).sum());score=(bad==0,retfit,full,pt*pt*(pad+.01)*nf,pt,pad,nf)
    if best is None or score>best[0]:best=(score,p)
 return best[1]
def fit_two(P,N):
 P=np.asarray(P);assert len(P)>=4
 # deterministic farthest-pair initialization and Lloyd assignment
 D=np.linalg.norm(P[:,None,:]-P[None,:,:],axis=2);i,j=np.unravel_index(np.argmax(D),D.shape);cent=np.array([P[i],P[j]])
 for _ in range(20):
  lab=np.argmin(np.linalg.norm(P[:,None,:]-cent[None,:,:],axis=2),axis=1)
  if min(np.bincount(lab,minlength=2))<2:lab=np.arange(len(P))%2
  new=np.array([P[lab==k].mean(0) for k in range(2)])
  if np.allclose(new,cent):break
  cent=new
 best=None
 for pt in [2.,1.75,1.5,1.25,1.,.9,.75,.6,.5,.4,.3]:
  for pad in [.15,.10,.075,.05,.025]:
   for nf in [1.,.85,.70,.55,.40]:
    cs=[fit_one('ncb_affine',P[lab==k],N,pt,pad,nf) for k in range(2)];p={'kind':'two_lobe','components':cs,'tan_expand':pt,'normal_pad':pad,'normal_factor':nf}
    bad=int(contains(p,N,True).sum()) if len(N) else 0;full=int(contains(p,P,False).sum());retfit=int(contains(p,P,True).sum());score=(bad==0,retfit,full,pt*pt*(pad+.01)*nf,pt,pad,nf)
    if best is None or score>best[0]:best=(score,p)
 return best[1]

def fit_clipped(P,N):
 P=np.asarray(P);N=np.asarray(N).reshape(-1,3);planes=[]
 for n in N:
  p=P[np.argmin(np.linalg.norm(P-n,axis=1))];u=p-n;u=u/max(np.linalg.norm(u),1e-12);nv=float(n@u);pv=P@u
  for qq in (0.,.25,.5):
   hi=float(np.quantile(pv,qq))
   if hi>nv+1e-6:planes.append({'u':u.tolist(),'threshold':(nv+hi)/2})
 # exact deterministic deduplication
 uniq=[];seen=set()
 for h in planes:
  k=tuple(np.round(np.r_[h['u'],h['threshold']],10))
  if k not in seen:seen.add(k);uniq.append(h)
 planes=uniq
 combos=[[]]+[[h] for h in planes]
 for i in range(len(planes)):
  for j in range(i+1,len(planes)):combos.append([planes[i],planes[j]])
 best=None
 for pt in [2.,1.5,1.,.75,.5]:
  for pad in [.15,.075,.025]:
   for nf in [1.,.7,.4]:
    base=fit_one('ncb_affine',P,N,pt,pad,nf)
    for clips in combos:
     par={'kind':'clipped_ncb','base':base,'clips':clips,'tan_expand':pt,'normal_pad':pad,'normal_factor':nf}
     bad=int(contains(par,N,True).sum()) if len(N) else 0;retfit=int(contains(par,P,True).sum());full=int(contains(par,P,False).sum())
     score=(bad==0,retfit,full,-len(clips),pt*pt*(pad+.01)*nf)
     if best is None or score>best[0]:best=(score,par)
 return best[1]

def fit_rap(P,N):
 P=np.asarray(P);N=np.asarray(N).reshape(-1,3);c=P.mean(0);C=np.cov((P-c).T,bias=True) if len(P)>1 else np.eye(3);w,V=np.linalg.eigh(C);R=V[:,np.argsort(w)[::-1]];z=(P-c)@R;base_lo=z.min(0);base_hi=z.max(0);mid=(base_lo+base_hi)/2;half=np.maximum((base_hi-base_lo)/2,.025);best=None
 for et in [2.,1.5,1.25,1.,.85,.7,.55,.4]:
  for en in [2.,1.5,1.25,1.,.85,.7,.55,.4]:
   for pad in [.10,.05,.025]:
    sc=np.array([et,et,en]);hh=(half+pad)*sc;par={'kind':'rap','c':c.tolist(),'R':R.tolist(),'lo':(mid-hh).tolist(),'hi':(mid+hh).tolist(),'tan_expand':et,'normal_expand':en,'pad':pad}
    bad=int(contains(par,N,True).sum()) if len(N) else 0;retfit=int(contains(par,P,True).sum());full=int(contains(par,P,False).sum());score=(bad==0,retfit,full,float(np.prod(hh)),et,en,pad)
    if best is None or score>best[0]:best=(score,par)
 return best[1]

def fit_two_rap(P,N):
 P=np.asarray(P);D=np.linalg.norm(P[:,None,:]-P[None,:,:],axis=2);i,j=np.unravel_index(np.argmax(D),D.shape);cent=np.array([P[i],P[j]])
 for _ in range(20):
  lab=np.argmin(np.linalg.norm(P[:,None,:]-cent[None,:,:],axis=2),axis=1)
  if min(np.bincount(lab,minlength=2))<2:lab=np.arange(len(P))%2
  new=np.array([P[lab==k].mean(0) for k in range(2)])
  if np.allclose(new,cent):break
  cent=new
 return {'kind':'two_rap','components':[fit_rap(P[lab==k],N) for k in range(2)]}

def approximate_diameter(par,known,Q):
 z=Q[contains(par,Q,False)];z=np.vstack([z,known[contains(par,known,False)]]) if len(known) else z
 if len(z)<2:return 0.
 a=z[0]
 for _ in range(4):a=z[np.argmax(np.linalg.norm(z-a,axis=1))]
 return float(np.max(np.linalg.norm(z-a,axis=1)))

inv=rows(HERE/'exact_q64_inventory.csv');split={r['state_id']:r['split'] for r in inv};by=defaultdict(list)
for r in inv:
 z=norm([r['eta1'],r['eta2'],r['eta3']]);q=dict(r,z=z,B=str(r['B63']).lower()=='true');by[r['state_id']].append(q)
# Common deterministic domain cloud only for geometry/diameter/universal audits.
sob=qmc.Sobol(3,scramble=False).random_base2(17);lo=np.array([-1.1666666667,-1.,-.5]);hi=np.array([1.,1.,.5]);Q=lo+(hi-lo)*sob;Q=Q[domain(Q)]
families=sys.argv[1:] or ['ncb_affine','racs','rfse_22','rfse_42','rfse_44']
for fam in families:
 kind='rfse' if fam.startswith('rfse') else fam;ex=(int(fam[-2]),int(fam[-1])) if fam.startswith('rfse') else (4,2);d=(HERE/'synthesized_forms'/fam) if fam in ('clipped_ncb','rap','two_rap') else HERE/('rfse' if fam.startswith('rfse') else fam);d.mkdir(parents=True,exist_ok=True)
 params={};metrics=[];pout=[];neighbor=[]
 for sid,rr in by.items():
  if split[sid]=='test':continue
  fitrows=[r for r in rr if r['evidence_role']=='fit_positive'];holdrows=[r for r in rr if r['evidence_role']=='independent_positive']
  # TRAIN-only deterministic minimum-fit fallback; no VAL evidence is moved.
  if split[sid]=='train' and len(fitrows)<3:
   move=sorted(holdrows,key=lambda r:(r['eta1'],r['eta2'],r['eta3']))[:3-len(fitrows)];fitrows+=move;holdrows=[r for r in holdrows if r not in move]
  P=np.array([r['z'] for r in fitrows]);N=np.array([r['z'] for r in rr if not r['B']],float).reshape(-1,3);H=np.array([r['z'] for r in holdrows],float).reshape(-1,3)
  if len(P)<3 or (kind in ('two_lobe','two_rap') and len(P)<4):continue
  par=fit_two(P,N) if kind=='two_lobe' else (fit_two_rap(P,N) if kind=='two_rap' else (fit_clipped(P,N) if kind=='clipped_ncb' else (fit_rap(P,N) if kind=='rap' else fit_state(kind,P,N,ex))));params[sid]=par
  negret=int(contains(par,N,True).sum()) if len(N) else 0;fullrec=float(contains(par,H,False).mean()) if len(H) else math.nan;retrec=float(contains(par,H,True).mean()) if len(H) else math.nan
  diam=approximate_diameter(par,np.vstack([P,H]) if len(H) else P,Q);sep=False
  inside_known=np.vstack([P,H])[contains(par,np.vstack([P,H]),True)] if len(H) else P[contains(par,P,True)]
  if len(inside_known)>1:sep=np.max(np.linalg.norm(inside_known[:,None]-inside_known[None,:],axis=2))>=.15
  neigh=[r for r in rr if r['B'] and ('cross_transfer_' in r['sources'] or 'val_neighbor_target' in r['sources'])]
  ncov=float(np.mean([contains(par,r['z'][None],False)[0] for r in neigh])) if neigh else math.nan
  metrics.append({'state_id':sid,'split':split[sid],'fit_B63':len(P),'independent_B63':len(H),'hard_negative':len(N),'retained_false_inclusion':negret,'full_B63_recall':fullrec,'retained_B63_recall':retrec,'diameter':diam,'retained_two_robust_separated':sep,'nondegenerate':diam>=.30 or sep})
  neighbor.append({'state_id':sid,'split':split[sid],'neighbor_B63':len(neigh),'full_neighbor_target_coverage':ncov})
  pout.append({'candidate_variant':fam,'state_id':sid,'split':split[sid],'parameters_json':json.dumps(par,separators=(',',':'))})
 # empirical universal max on Q plus all observed eta and model centers
 train=[s for s in params if split[s]=='train'];cand=[Q[::8]]
 cand.append(np.vstack([r['z'] for sid,rr in by.items() if split[sid]=='train' for r in rr]))
 C=np.vstack(cand);cov=np.array([sum(contains(params[s],C,True) for s in train)]).reshape(-1) if False else np.sum(np.stack([contains(params[s],C,True) for s in train]),axis=0)
 umax=int(cov.max()) if len(cov) else 0;uarg=C[int(np.argmax(cov))].tolist() if len(cov) else None
 val=[r for r in metrics if r['split']=='val'];nv=[r for r in neighbor if r['split']=='val' and not math.isnan(r['full_neighbor_target_coverage'])]
 full=[r['full_B63_recall'] for r in val if not math.isnan(r['full_B63_recall'])];ret=[r['retained_B63_recall'] for r in val if not math.isnan(r['retained_B63_recall'])]
 gate={'candidate_variant':fam,'usable_val_states':len(val),'cached_retained_false_inclusion':sum(r['retained_false_inclusion'] for r in val),
  'val_median_full_B63_recall':float(np.median(full)) if full else None,'val_states_full_recall_ge_0_40':sum(r['full_B63_recall']>=.4 for r in val if not math.isnan(r['full_B63_recall'])),
  'val_median_retained_B63_recall':float(np.median(ret)) if ret else None,'val_nondegenerate_states':sum(r['nondegenerate'] for r in val),
  'val_median_neighbor_target_coverage':float(np.median([r['full_neighbor_target_coverage'] for r in nv])) if nv else None,
  'train_universal_retained_max_count':umax,'train_sets':len(train),'train_universal_retained_max_fraction':umax/len(train) if train else None,'train_universal_eta_normalized':uarg,
  'fresh_validation_status':'PENDING','cached_geometry_gate_pass':False}
 gate['cached_geometry_gate_pass']=(len(val)==8 and gate['cached_retained_false_inclusion']==0 and gate['val_median_full_B63_recall'] is not None and gate['val_median_full_B63_recall']>=.60 and gate['val_states_full_recall_ge_0_40']>=6 and gate['val_median_retained_B63_recall']>=.35 and gate['val_nondegenerate_states']>=6 and gate['val_median_neighbor_target_coverage'] is not None and gate['val_median_neighbor_target_coverage']>=.60 and gate['train_universal_retained_max_fraction']<=.75)
 write(d/'fitted_parameters.csv',pout);write(d/f'cached_precision_recall_{fam}.csv',metrics);write(d/f'neighbor_target_coverage_{fam}.csv',neighbor)
 (d/f'gate_{fam}.json').write_text(json.dumps(gate,indent=2)+'\n');(d/f'complexity_{fam}.json').write_text(json.dumps({'family':fam,'scalar_parameters_per_state':24 if kind=='two_rap' else (12 if kind=='rap' else (19 if kind=='clipped_ncb' else (11 if kind=='ncb_affine' else (20 if kind=='racs' else 12)))),'membership':'finite analytic inequalities plus E_bridge halfspaces','erosion':{'gamma_tan':GT,'gamma_norm':GN,'clip_margin':.025 if kind=='clipped_ncb' else None}},indent=2)+'\n')
 print(json.dumps(gate,indent=2))
