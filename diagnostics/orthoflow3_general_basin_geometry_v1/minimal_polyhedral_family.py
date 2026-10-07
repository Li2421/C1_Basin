#!/usr/bin/env python3
from audit import *
from scipy.spatial import ConvexHull
from scipy.optimize import milp,Bounds,LinearConstraint
def fit_minimal(P,N):
 from families import fit
 facets=np.unique(ConvexHull(P).equations,axis=0);reject=N@facets[:,:3].T+facets[:,3]>1e-9 if len(N) else np.zeros((0,len(facets)),bool);external=reject.any(axis=1);chosen=[]
 if external.any():
  cover=reject[external].astype(float);loss=np.mean(P@facets[:,:3].T+facets[:,3]>-.025,axis=0)
  objective=1+1e-5*loss+1e-9*np.arange(len(facets))/max(1,len(facets))
  sol=milp(objective,integrality=np.ones(len(facets)),bounds=Bounds(0,1),constraints=LinearConstraint(cover,1,np.inf),options={'time_limit':20,'mip_rel_gap':0})
  assert sol.status==0,('minimum facet selection unresolved',sol.message)
  chosen=np.flatnonzero(sol.x>.5).tolist();assert len(chosen)<=8,'Frozen eight-face capacity exceeded'
  assert np.all(cover[:,chosen].sum(axis=1)>=1)
 remaining=~external;cut=fit(P,N[remaining],'semialgebraic') if remaining.any() else {'w':[1.]+[0.]*9,'optimization_success':True}
 return dict(family='minimal_polyhedral_support_quadratic_exclusion',planes=facets[chosen].tolist(),w=cut['w'],caps=[],parameter_count=4*len(chosen)+(10 if remaining.any() else 0),meaningful_parameter_count=3*len(chosen)+(9 if remaining.any() else 0),
  optimization_success=cut['optimization_success'],minimum_support_facets=len(chosen),negatives_excluded_by_support=int(external.sum()),negative_quadratic_constraints=int(remaining.sum()),fit_positive_count=len(P))
if __name__=='__main__':
 from families import run
 run(sys.argv[1] if len(sys.argv)>1 else 'minimal_polyhedral_r5_corrected',True,kinds=['minimal_polyhedral_support_quadratic_exclusion'])
