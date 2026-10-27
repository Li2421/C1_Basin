"""Actual product normals and a declared coupled reference in stage-two units."""
from experiment import *
import numpy as np
from stage2_safety import acceleration_problem,unicycle_matrix,unicycle_next,unicycle_q,LOOKAHEAD
from shared_control.hard_projection import HardProjectionConfig

def local_rows(z,t,i,c,item):
    rows=[list(row) for row in item['A']];kinds=[list(k) for k in item['kinds']]
    if c.scene=='lit_doorway':
        inverse=np.linalg.inv(unicycle_matrix(z['headings'][t,i]));control=inverse@z['safe'][t,i]
        for k,(lower,upper) in enumerate(((-.26,.26),(-3.8,3.8))):
            if control[k]-lower<=1e-7:rows.append(inverse[k].tolist());kinds.append(['input_lower',i,k])
            if upper-control[k]<=1e-7:rows.append((-inverse[k]).tolist());kinds.append(['input_upper',i,k])
    return np.asarray(rows,float).reshape(-1,2),kinds

def doorway_pair_value_and_normal(p,heading,front,i,j,c):
    q=[];future=[];jac=[]
    for k in (i,j):
        theta=heading[k];T=unicycle_matrix(theta);control=np.linalg.solve(T,front[k]);pn,hn=unicycle_next(p[k],theta,control,c.dt)
        q.append(unicycle_q(p[k],theta));future.append(unicycle_q(pn,hn))
        J=np.column_stack([c.dt*np.asarray([np.cos(theta),np.sin(theta)]),LOOKAHEAD*c.dt*np.asarray([-np.sin(hn),np.cos(hn)])])
        jac.append(J@np.linalg.inv(T))
    radius=2*(c.agent_radius+LOOKAHEAD)+c.agent_collision_margin+HardProjectionConfig().separation_buffer
    delta=future[0]-future[1];old=q[0]-q[1]
    value=float(delta@delta-radius*radius-.9*(old@old-radius*radius))
    row=np.zeros((len(p),2));row[i]=2*delta@jac[0];row[j]=-2*delta@jac[1]
    return value,row.ravel()

def coupled_active_rows(z,t,c,wall_segments,local):
    p=z['positions'][t];v=z['velocities'][t];safe=z['safe'][t];n=len(p);rows=[]
    if c.scene=='lit_doorway':
        for i in range(n):
            for j in range(i+1,n):
                value,row=doorway_pair_value_and_normal(p,z['headings'][t],safe,i,j,c)
                if value<=1e-7:rows.append(row)
        for i,(A,kinds) in enumerate(local):
            for a,kind in zip(A,kinds):
                if kind[0]=='pair':continue
                row=np.zeros((n,2));row[i]=a;rows.append(row.ravel())
    else:
        for i in range(n):
            A,lower,kinds=acceleration_problem(p,v,i,c,wall_segments,**({k:v for k,v in json.loads(str(z['safety_fix_json'])).items() if k in ('gamma','circle_discrete')} if 'safety_fix_json' in z else {}))
            for a,b,kind in zip(A,lower,kinds):
                row=np.zeros((n,2));row[i]=a
                if kind[0]=='pair':
                    j=kind[2]
                    if i>j:continue
                    params={k:v for k,v in json.loads(str(z['safety_fix_json'])).items() if k in ('gamma','circle_discrete')} if 'safety_fix_json' in z else {}
                    Aj,bj,kj=acceleration_problem(p,v,j,c,wall_segments,**params)
                    partner=next(k for k,label in enumerate(kj) if label[0]=='pair' and label[2]==i)
                    row[j]=Aj[partner];b=b+bj[partner]
                if row.ravel()@safe.ravel()-b<=1e-7:rows.append(row.ravel())
            for a,kind in zip(*local[i]):
                if kind[0]=='acceleration_bound':
                    row=np.zeros((n,2));row[i]=a;rows.append(row.ravel())
    return np.asarray(rows,float).reshape(-1,2*n)
