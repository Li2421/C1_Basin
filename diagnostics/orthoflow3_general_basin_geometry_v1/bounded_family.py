#!/usr/bin/env python3
from audit import *
from scipy.spatial import ConvexHull
def fit_bounded(P,N):
 import cvxpy as cp
 from families import polynomial,fit
 hull=ConvexHull(P);in_hull=np.all(N@hull.equations[:,:3].T+hull.equations[:,3]<=1e-9,axis=1) if len(N) else np.zeros(0,bool)
 X=polynomial(P);Y=polynomial(N[~in_hull]);w=cp.Variable(10)
 A=cp.bmat([[w[4],w[7]/2,w[8]/2],[w[7]/2,w[5],w[9]/2],[w[8]/2,w[9]/2,w[6]]])
 constraints=[A << -.001*np.eye(3),w>=-50,w<=50]
 if len(Y):constraints.append(Y@w<=-.01)
 prob=cp.Problem(cp.Minimize(cp.sum_squares(cp.pos(1-X@w))/len(P)+.002*cp.sum_squares(w[1:])),constraints)
 prob.solve(solver='CLARABEL',max_iter=1000,tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9)
 assert w.value is not None,(prob.status,'no bounded support fit')
 base=np.array(w.value);quad=np.array([[base[4],base[7]/2,base[8]/2],[base[7]/2,base[5],base[9]/2],[base[8]/2,base[9]/2,base[6]]])
 feasible=(not len(Y) or np.max(Y@base)<=-.01+1e-7) and np.linalg.eigvalsh(quad).max()<=-.001+1e-7
 assert feasible,'Bounded support optimizer violated frozen constraints'
 cut=fit(P,N[in_hull],'semialgebraic') if in_hull.any() else {'w':[1.]+[0.]*9,'optimization_success':True}
 return dict(family='bounded_quadratic_with_exclusion',quadrics=[base.tolist(),cut['w']],parameter_count=20 if in_hull.any() else 10,
  support_status=prob.status,optimization_success=bool(feasible and cut['optimization_success']),exterior_negatives=int((~in_hull).sum()),interior_negatives=int(in_hull.sum()),caps=[])
if __name__=='__main__':
 from families import run
 run(sys.argv[1] if len(sys.argv)>1 else 'bounded_r2',True,kinds=['bounded_quadratic_with_exclusion'])
