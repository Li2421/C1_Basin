"""Prerequisite-stratified gate; all inputs/out-of-premise cases are retained."""
import bootstrap
from experiment import *
from swap_geometry import config
from swap_safety_user_fix_20261023 import SwapSafetyRevision,GAMMAS
from single_integrator.environment import point_segment_distance
import numpy as np,time

def snap(p,o):return dict(positions=np.asarray(p).reshape(1,2),moving_position=np.asarray(o),walls=np.zeros((0,2,2)),config=config('lit_swap8_moving',1).to_dict())
def main():
    assert not (HERE/'MOVING_USER_FIX_UNIT_GATE.json').exists();D=.35+.01*.1001;dt=.05;physical=.35;headon=[]
    gaps=np.r_[np.linspace(0,.8,161),1.2,2.,3.]
    for gamma in GAMMAS:
        minimum=1e9;min_D_gap=1e9;bad=[];max_residual=0.
        for gap in gaps:
            p=np.zeros(2);o=np.array([-D-gap,0.])
            for step in range(200):
                f=SwapSafetyRevision(snap(p,o),0,gamma);a,d=f.project([[-.5,0.]]);u=a[0];vo=np.array([.25,0.]);distance=float(point_segment_distance(np.zeros(2),p-o,p-o+dt*(u-vo)))
                minimum=min(minimum,distance);min_D_gap=min(min_D_gap,distance-D);max_residual=max(max_residual,d['maximum_violation'])
                if distance<=physical or d['restoration_phase']!='original':bad.append(dict(gap=float(gap),step=step,distance=distance,certificate=d));break
                p=p+dt*u;o=o+dt*vo
        r=dict(gamma=gamma,passed=not bad,initial_gaps=len(gaps),steps_per_gap=200,minimum_center_distance=minimum,minimum_inflated_gap=min_D_gap,maximum_row_violation=max_residual,failures=bad);headon.append(r)
        write(HERE/'MOVING_USER_FIX_HEAD_ON_PROGRESS.json',headon);print('Head-on',gamma,'passed',not bad,'minimum',minimum,flush=True)
    eligible=[r['gamma'] for r in headon if r['passed']];assert eligible
    gamma=max(eligible);rng=np.random.default_rng(2026102321);N=10000
    angles=rng.uniform(0,2*np.pi,N);normal=np.stack([np.cos(angles),np.sin(angles)],-1);gaps=rng.uniform(-.2,1.5,N);origins=rng.uniform(-2,2,(N,2));obs=origins-(D+gaps)[:,None]*normal
    vangle=rng.uniform(0,2*np.pi,N);speed=rng.uniform(0,.5,N);targets=speed[:,None]*np.stack([np.cos(vangle),np.sin(vangle)],-1);targets[:5000]=0
    np.savez_compressed(HERE/'MOVING_USER_FIX_SNAPSHOTS.npz',positions=origins,obstacles=obs,nominal=targets,epsilon=.001001,gamma=gamma)
    counts=dict(total=N,premise=0,outside_premise=0,premise_failures=0,unknown=0);outside=[];failures=[];minimum=1e9
    for k in range(N):
        p=origins[k];o=obs[k];f=SwapSafetyRevision(snap(p,o),0,gamma);actual_gap=np.linalg.norm(p-o)-D;premise=actual_gap>=0;counts['premise' if premise else 'outside_premise']+=1
        if not premise:outside.append(k)
        try:
            a,cert=f.project(targets[k:k+1]);vo=.25*(p-o)/np.linalg.norm(p-o);distance=float(point_segment_distance(np.zeros(2),p-o,p-o+dt*(a[0]-vo)))
            if premise:
                minimum=min(minimum,distance)
                if distance<=physical or distance<D-1e-7 or cert['restoration_phase']!='original':counts['premise_failures']+=1;failures.append(dict(index=k,gap=actual_gap,distance=distance,certificate=cert))
        except Exception as exc:counts['unknown']+=1;failures.append(dict(index=k,premise=premise,error=str(exc)))
    passed=counts['premise_failures']==0 and counts['unknown']==0
    write(HERE/'MOVING_USER_FIX_UNIT_GATE.json',dict(utc=utc(),passed=passed,selected_gamma=gamma,gamma_grid=list(GAMMAS),head_on=headon,random_snapshots=counts,outside_premise_indices=outside,failures=failures,minimum_premise_center_distance=minimum,physical_pair_radius=physical,inflated_distance=D,epsilon=.001001,solver_geometric_certificate_tolerance=1e-7,snapshot_sha256=sha(HERE/'MOVING_USER_FIX_SNAPSHOTS.npz'),rule='Only g>=0 and original feasible snapshots gate safety; outside cases retained; original unknown policy preserved'))
    assert passed,counts;print('Moving user-fix unit gate PASSED gamma',gamma,counts,flush=True)
if __name__=='__main__':main()
