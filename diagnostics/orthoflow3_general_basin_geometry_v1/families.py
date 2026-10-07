#!/usr/bin/env python3
"""Small analytic geometric hypotheses. No policy/critic fitting."""
from audit import *
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import pdist
def cluster(X,k):
 centers=[X[0]]
 while len(centers)<min(k,len(X)):centers.append(X[np.argmax(cdist(X,centers).min(axis=1))])
 C=np.array(centers)
 for _ in range(40):
  lab=cdist(X,C).argmin(axis=1);new=np.array([X[lab==j].mean(axis=0) if np.any(lab==j) else C[j] for j in range(len(C))])
  if np.max(np.abs(new-C))<1e-12:break
  C=new
 return [X[lab==j] for j in range(len(C)) if np.any(lab==j)]
def boundary_caps(P,N,maxk=4):
 if len(N)==0:return []
 choices=[]
 for k in range(1,min(maxk,len(N))+1):
  caps=[]
  for group in cluster(N,k):
   mean=group.mean(axis=0);opts=[]
   for face in HS:
    normal=face[:3];base=mean-(mean@normal+face[3])*normal
    # Centers on/exterior to a domain facet: each removed ball opens onto exterior.
    for shift in (0.,.25,.5,1.):
     c=base+shift*normal;r=np.linalg.norm(group-c,axis=1).max()+.002
     survive=np.linalg.norm(P-c,axis=1)>=r+.025
     opts.append((survive.sum(),-r,c,r))
   best=max(opts,key=lambda a:a[:2]);caps.append({'c':best[2].tolist(),'r':float(best[3])})
  kept=np.ones(len(P),bool)
  for cap in caps:kept &= np.linalg.norm(P-cap['c'],axis=1)>=cap['r']+.025
  choices.append((float(kept.mean())-.015*k,-k,caps))
 return max(choices,key=lambda a:a[:2])[2]
def polynomial(X):
 X=np.atleast_2d(X);x,y,z=X.T
 return np.column_stack([np.ones(len(X)),x,y,z,x*x,y*y,z*z,x*y,x*z,y*z])
def base_model(P,kind):
 c=P.mean(axis=0);_,_,U=np.linalg.svd(P-c,full_matrices=False);R=U.T
 if kind in ('common_core_minus_exclusions','semialgebraic'):return {}
 if kind=='conditional_band_with_cuts':
  x=P[:,:2];mu=x.mean(axis=0);cov=np.cov(x.T)+1e-5*np.eye(2);A=np.linalg.inv(cov);A/=np.max(np.einsum('ni,ij,nj->n',x-mu,A,x-mu))+1e-10
  X=np.column_stack([np.ones(len(P)),x-mu]);bounds=[(None,None),(-3,3),(-3,3)]
  lower=linprog([-1,0,0],A_ub=X,b_ub=P[:,2],bounds=bounds,method='highs')
  upper=linprog([1,0,0],A_ub=-X,b_ub=-P[:,2],bounds=bounds,method='highs')
  assert lower.success and upper.success
  return {'mu':mu.tolist(),'A':A.tolist(),'lower':lower.x.tolist(),'upper':upper.x.tolist()}
 if kind=='asymmetric_slab_with_notches':
  Z=(P-c)@R;lo=Z.min(axis=0)-.002;hi=Z.max(axis=0)+.002
  return {'c':c.tolist(),'R':R.tolist(),'lo':lo.tolist(),'hi':hi.tolist()}
 if kind=='star_convex_with_exclusions':
  # Positive definite ellipsoidal radial body, centered at an observed robust medoid.
  c=P[np.argmin(cdist(P,P).sum(axis=1))];cov=np.cov((P-c).T)+1e-4*np.eye(3);A=np.linalg.inv(cov);A/=np.max(np.einsum('ni,ij,nj->n',P-c,A,P-c))+1e-10
  return {'c':c.tolist(),'A':A.tolist()}
def fit(P,N,kind):
 if kind=='minimal_polyhedral_support_quadratic_exclusion':
  from minimal_polyhedral_family import fit_minimal
  return fit_minimal(P,N)
 if kind=='polyhedral_support_quadratic_exclusion':
  from polyhedral_family import fit_polyhedral
  return fit_polyhedral(P,N)
 if kind=='bounded_quadratic_with_exclusion':
  from bounded_family import fit_bounded
  return fit_bounded(P,N)
 model={'family':kind,**base_model(P,kind)}
 if kind=='semialgebraic':
  X=polynomial(P);Y=polynomial(N);initial=np.zeros(10);initial[0]=-1 if len(N) else 1
  def fun(w):
   r=np.minimum(X@w-1,0);return np.mean(r*r)+.002*np.dot(w[1:],w[1:])
  def jac(w):
   r=np.minimum(X@w-1,0);j=2*X.T@r/len(X);j[1:]+=.004*w[1:];return j
  constraints=[] if not len(N) else [{'type':'ineq','fun':lambda w:-.01-Y@w,'jac':lambda w:-Y}]
  sol=minimize(fun,initial,jac=jac,constraints=constraints,method='SLSQP',bounds=[(-50,50)]*10,options={'maxiter':600,'ftol':1e-10})
  model.update(w=sol.x.tolist(),optimization_success=bool(sol.success),optimization_message=sol.message,caps=[])
 else:
  # No cap needs to remove a negative already outside the base geometry.
  mask=contains(model|{'caps':[]},N) if len(N) else []
  model['caps']=boundary_caps(P,N[mask],4 if kind=='common_core_minus_exclusions' else 2)
 model['parameter_count']=10 if kind=='semialgebraic' else 4*len(model['caps'])+{'common_core_minus_exclusions':0,'conditional_band_with_cuts':11,'asymmetric_slab_with_notches':12,'star_convex_with_exclusions':9}[kind]
 return model
def contains(m,Z,ret=False):
 Z=np.atleast_2d(Z);kind=m['family'];ok=inside(Z);delta=.025 if ret else 0.
 if ret:ok &= np.max(Z@HS[:,:3].T+HS[:,3],axis=1)<=-delta+1e-10
 if kind in ('polyhedral_support_quadratic_exclusion','minimal_polyhedral_support_quadratic_exclusion'):
  planes=np.array(m['planes']).reshape(-1,4)
  if len(planes):ok &= np.all(Z@planes[:,:3].T+planes[:,3]<=-delta+1e-10,axis=1)
 if kind=='conditional_band_with_cuts':
  x=Z[:,:2]-m['mu'];A=np.array(m['A']);q=np.einsum('ni,ij,nj->n',x,A,x);ok &= q <=(.85**2 if ret else 1)+1e-10
  X=np.column_stack([np.ones(len(Z)),x]);low=X@m['lower'];high=X@m['upper'];mid=(low+high)/2;rad=(high-low)/2*(.75 if ret else 1)
  ok &= (Z[:,2]>=mid-rad-1e-10)&(Z[:,2]<=mid+rad+1e-10)&(rad>=0)
 elif kind=='asymmetric_slab_with_notches':
  Y=(Z-m['c'])@np.array(m['R']);lo=np.array(m['lo']);hi=np.array(m['hi']);mid=(lo+hi)/2;rad=(hi-lo)/2*(np.array([.85,.85,.75]) if ret else 1)
  ok &= np.all(abs(Y-mid)<=rad+1e-10,axis=1)
 elif kind=='star_convex_with_exclusions':
  Y=Z-m['c'];A=np.array(m['A']);ok &= np.einsum('ni,ij,nj->n',Y,A,Y)<=(.75**2 if ret else 1)+1e-10
 elif kind in ('semialgebraic','bounded_quadratic_with_exclusion','polyhedral_support_quadratic_exclusion','minimal_polyhedral_support_quadratic_exclusion'):
  for coefficients in (m['quadrics'] if kind=='bounded_quadratic_with_exclusion' else [m['w']]):
   w=np.array(coefficients);A=np.array([[w[4],w[7]/2,w[8]/2],[w[7]/2,w[5],w[9]/2],[w[8]/2,w[9]/2,w[6]]]);val=polynomial(Z)@w
   penalty=delta*np.linalg.norm(2*Z@A+w[1:4],axis=1)+delta**2*np.linalg.norm(A,2)
   ok &= val>=penalty-1e-10
 for cap in m.get('caps',[]):ok &= np.linalg.norm(Z-cap['c'],axis=1)>=cap['r']+delta-1e-10
 return ok
def run(tag='initial',all_known_negatives=False,kinds=None,reuse_tag=None):
 if (HERE/f'family_models_{tag}.json').exists():raise FileExistsError('Fit tag already frozen; use a new tag instead of overwriting scientific history')
 inv=read(HERE/'exact_q64_inventory.csv');rows=[];models=[]
 write(f'fitting_evidence/{tag}.csv',inv)
 dump(f'fitting_evidence/{tag}_provenance.json',{'inventory_sha256':sha(HERE/'exact_q64_inventory.csv'),'script_sha256':sha(Path(__file__)),'snapshot':f'fitting_evidence/{tag}.csv'})
 development=json.load(open(HERE/'development_validation_batches.json'))['batches'] if (HERE/'development_validation_batches.json').exists() else []
 reusable={};reuse_evidence=[]
 if reuse_tag:
  reusable={(r['state_id'],r['model']['family']):r['model'] for r in json.load(open(HERE/f'family_models_{reuse_tag}.json'))}
  reuse_evidence=read(HERE/f'fitting_evidence/{reuse_tag}.csv')
 kinds=kinds or ['common_core_minus_exclusions','conditional_band_with_cuts','asymmetric_slab_with_notches','star_convex_with_exclusions','semialgebraic']
 for sid in sorted(set(r['state_id'] for r in inv)):
  rr=[r for r in inv if r['state_id']==sid];Z=np.array([(eta(r)-AFF)/SCALE for r in rr]);pos=np.array([r['B63']=='True' for r in rr]);primary=np.array(['retained_validation' not in r['phases'] for r in rr]);fitmask=primary&np.array([r['geometry_role']=='fit' for r in rr]);ho=primary&~fitmask
  fitmask |= np.array([any(b in r['phases'].split(';') for b in development) for r in rr])
  adequate=len(rr)>=30 and pos.sum()>=10 and (ho&pos).sum()>=4 and (~pos).sum()>=3
  if not adequate:continue
  P=Z[pos&fitmask];N=Z[~pos] if all_known_negatives else Z[~pos&fitmask]
  ind=np.array(['coverage64' in r['phases'] or 'coverage_screen' in r['phases'] for r in rr])&pos
  transfer=np.array(['cross_transfer' in r['sources'] or 'common_core' in r['phases'] or 'val_neighbor' in r['phases'] for r in rr])&pos
  for kind in kinds:
   previous=reusable.get((sid,kind));reuse=False
   if previous and all_known_negatives:
    from audit_validation_reuse import input_keys
    reuse=input_keys(reuse_evidence,sid,previous)==input_keys(inv,sid,{'development_validation_batches':development})
   m=dict(previous) if reuse else fit(P,N,kind)
   if reuse:m['unchanged_input_instance_reused_from']=reuse_tag
   m['fit_negative_policy']='ALL_KNOWN_EXACT_NEGATIVES' if all_known_negatives else 'HASH_FIT_NEGATIVES';m['development_validation_batches']=development;full=contains(m,Z);ret=contains(m,Z,True)
   row=dict(state_id=sid,family=kind,n=len(rr),fit_B63=int((fitmask&pos).sum()),heldout_B63=int((ho&pos).sum()),
    full_recall=float(full[ho&pos].mean()),retained_recall=float(ret[ho&pos].mean()),cached_false_inclusion=int((ret&~pos).sum()),cached_full_false_inclusion=int((full&~pos).sum()),heldout_full_false_inclusion=int((full&~pos&ho).sum()),
    independent_coverage_recall=float(full[ind].mean()) if ind.any() else '',transfer_recall=float(full[transfer].mean()) if transfer.any() else '',transfer_B63_count=int(transfer.sum()),transfer_captured_count=int((full&transfer).sum()),parameters=m['parameter_count'],meaningful_parameters=m.get('meaningful_parameter_count',m['parameter_count']))
   rows.append(row);models.append(dict(state_id=sid,model=m));dump(f'{kind}/{tag}/{sid}.json',m)
 write(f'family_metrics_{tag}.csv',rows);dump(f'family_models_{tag}.json',models)
 gates=[]
 for kind in kinds:
  rr=[r for r in rows if r['family']==kind];n=len(rr)
  if not n:continue
  recall=float(np.median([r['full_recall'] for r in rr]));ret=float(np.median([r['retained_recall'] for r in rr]));frac=float(np.mean([r['full_recall']>=.5 for r in rr]));neg=sum(r['cached_false_inclusion'] for r in rr)
  tr=[r['transfer_recall'] for r in rr if r['transfer_recall']!=''];t=float(np.mean(tr)) if tr else None
  transfer_count=sum(r['transfer_B63_count'] for r in rr);pooled=sum(r['transfer_captured_count'] for r in rr)/transfer_count if transfer_count else None
  full_negative=sum(r['cached_full_false_inclusion'] for r in rr)
  gate=dict(family=kind,adequate_states=n,median_full_recall=recall,median_retained_recall=ret,fraction_recall_ge_half=frac,cached_retained_false_inclusions=neg,mean_state_transfer_recall=t,
   pooled_transfer_recall=pooled,cached_full_false_inclusions=full_negative,
   cached_screen_pass=recall>=.7 and ret>=.4 and frac>=.8 and neg==0 and full_negative==0 and pooled is not None and pooled>=.7,fresh_retained_validation='NOT_RUN',generalized_gate_pass=False)
  gates.append(gate)
 dump(f'family_gates_{tag}.json',gates)
 print(json.dumps({'fit_tag':tag,'adequate_states':len(set(r['state_id'] for r in rows)),'passing_cached_families':sum(g['cached_screen_pass'] for g in gates),'details':f'family_gates_{tag}.json'}))
if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'initial','--all-known-negatives' in sys.argv)
