"""Maximal arm length from geometry alone, including unchanged perturbed states."""
import bootstrap
from experiment import *
import numpy as np
from scipy.optimize import linprog
from single_integrator.environment import segment_distance

def main():
    rows=[];bounds=[];witness=[];directions=[];clearance=.05+.005+.0001+1e-6;corridor=.15+1e-6
    states=read(HERE/'FLOW_STATES_lit_swap8_obstacles.json')
    for theta in np.deg2rad([22.5,112.5,202.5,292.5]):
        e=np.array([np.cos(theta),np.sin(theta)]);f=np.array([-e[1],e[0]])
        for side in (-1,1):
            d=(-e+side*f)/np.sqrt(2);directions.append((e,d))
            for endpoint in (False,True):
                # Infinite-line corridors: each entire arm is the convex hull of endpoints.
                for angle in np.arange(4)*np.pi/4:
                    n=np.array([-np.sin(angle),np.cos(angle)]);sign=np.sign(n@e);coeff=np.array([sign*(n@e),sign*(n@d) if endpoint else 0.])
                    rows.append(-coeff);bounds.append(-corridor);witness.append(dict(kind='nominal_corridor',theta=theta,side=side,endpoint=endpoint,line_angle=angle))
                # Straight center trajectories plus original radius/wall safety clearance.
                for state in states:
                    for i,(a,b) in enumerate(zip(np.asarray(state['positions']),np.asarray(state['goals']))):
                        v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v);sign=np.sign(n@(.9*e-a));assert sign and np.sign(n@(.7*e-a))==sign and np.sign(n@(1.1*e-a))==sign
                        coeff=np.array([sign*(n@e),sign*(n@d) if endpoint else 0.]);rhs=clearance+sign*(n@a)
                        rows.append(-coeff);bounds.append(-rhs);witness.append(dict(kind='existing_state_straight_clearance',state=state['index'],agent=i,theta=theta,side=side,endpoint=endpoint))
    fit=linprog([0.,-1.],A_ub=np.asarray(rows),b_ub=np.asarray(bounds),bounds=[(.7,1.1),(0.,None)],method='highs');assert fit.success,fit.message
    R,L=map(float,fit.x);slacks=np.asarray(bounds)-np.asarray(rows)@fit.x;active=np.flatnonzero(slacks<1e-8)
    walls=np.array([[R*e,R*e+L*d] for e,d in directions]);corr=[]
    for angle in np.arange(4)*np.pi/4:
        n=np.array([-np.sin(angle),np.cos(angle)]);values=walls@n;assert np.all(values[:,0]*values[:,1]>0);corr.append(float(np.min(abs(values))))
    minimum=1e9;worst=None;grid=[]
    for state in states:
        p=np.asarray(state['positions']);g=np.asarray(state['goals']);dist=segment_distance(p[:,None],g[:,None],walls[None,:,0],walls[None,:,1]);i,j=np.unravel_index(np.argmin(dist),dist.shape)
        # Every wall endpoint projects into the actual finite start-goal segment.
        v=g-p;projection=np.sum((walls[None,:,:,:]-p[:,None,None,:])*v[:,None,None,:],-1)/np.sum(v*v,-1)[:,None,None]
        assert np.all((projection>0)&(projection<1))
        if dist[i,j]<minimum:minimum=float(dist[i,j]);worst=dict(state=state['index'],agent=int(i),wall=int(j))
        assert dist.min()>=.0551-1e-9
        grid.append(dict(state=state['index'],straight_path_minimum_center_wall_distance=dist.min(1),physical_wall_clearance=dist.min(1)-.055,safety_wall_clearance=dist.min(1)-.0551,all_straight_paths_valid=True))
    out=dict(utc=utc(),corner_radius=R,arm_length=L,walls=walls,angles_degrees=[22.5,112.5,202.5,292.5],notch_orientation='Inward radial; arms (-e_r +/- e_t)/sqrt(2)',corridor_halfwidth=.15,nominal_corridor_minimum_distances=corr,
      straight_minimum_center_wall_distance=minimum,worst_straight_path=worst,wall_physical_margin=.055,wall_safety_margin=.0551,design_numerical_clearance=1e-6,
      nominal_corridor_only_supremum=(1.1*np.sin(np.pi/8)-.15)/np.cos(np.pi/8),optimization='2-variable linear program maximize common L; R in[.7,1.1], fixed orientation; nominal corridors and unchanged 32 physical straight paths only; no outcomes or capability labels used',
      active_constraints=[witness[i] for i in active],lp_dual_inequality_marginals=fit.ineqlin.marginals,lp_status=fit.message,maximum_linear_constraint_violation=float(np.maximum(-slacks,0).max()),states=grid)
    write(HERE/'LAYOUT_REVISION_DESIGN.json',out);print('Revised layout R=',R,'L=',L,'nominal corridor min',min(corr),'straight min',minimum,flush=True)
if __name__=='__main__':main()
