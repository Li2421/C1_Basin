#!/usr/bin/env python3
from audit import *
from scipy.spatial import ConvexHull
def fit_polyhedral(P,N):
 from families import fit
 facets=np.unique(ConvexHull(P).equations,axis=0);remaining=np.ones(len(N),bool);retained=np.ones(len(P),bool);chosen=[]
 for _ in range(3):
  options=[]
  for i,f in enumerate(facets):
   if i in chosen:continue
   reject=N@f[:3]+f[3]>1e-9 if len(N) else np.zeros(0,bool);new=int((reject&remaining).sum())
   if not new:continue
   new_retained=P@f[:3]+f[3]<=-.025;lost=int((retained&~new_retained).sum());options.append((new-.05*lost,new,-lost,-i,i,reject,new_retained))
  if not options:break
  best=max(options,key=lambda x:x[:4]);chosen.append(best[4]);remaining &= ~best[5];retained &= best[6]
 cut=fit(P,N[remaining],'semialgebraic') if remaining.any() else {'w':[1.]+[0.]*9,'optimization_success':True}
 return dict(family='polyhedral_support_quadratic_exclusion',planes=facets[chosen].tolist(),w=cut['w'],caps=[],parameter_count=4*len(chosen)+(10 if remaining.any() else 0),optimization_success=cut['optimization_success'],
  negatives_excluded_by_support=int(len(N)-remaining.sum()),negative_quadratic_constraints=int(remaining.sum()),fit_positive_count=len(P))
if __name__=='__main__':
 from families import run
 run('polyhedral_r4',True,kinds=['polyhedral_support_quadratic_exclusion'])
