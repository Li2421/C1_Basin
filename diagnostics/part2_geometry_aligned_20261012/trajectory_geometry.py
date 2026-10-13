"""Fixed seed0 early geometric trajectories from old complete rollout cache."""
import csv,json
from pathlib import Path
import numpy as np
from analyze_geometry import HERE,HARD,OLD,readcsv,writecsv,wrap

def ring(p,H):
    q=p[:min(H+1,len(p))]
    ang=np.unwrap(np.arctan2(q[:,:,1],q[:,:,0]),axis=0)
    net=ang[-1]-ang[0]
    traveled=np.sum(np.abs(np.diff(ang,axis=0)),axis=0)
    return net,traveled

def toy(p,H):
    q=p[:min(H+1,len(p))]
    d=np.linalg.norm(q[:,0]-q[:,1],axis=1)
    return dict(min_pair_distance=float(np.min(d)),
        max_y=float(np.max(q[:,:,1])),
        max_lateral_separation=float(np.max(np.abs(q[:,0,1]-q[:,1,1]))),
        max_y_inside_bottleneck=float(np.max(np.where(np.abs(q[:,:,0])<=.8,q[:,:,1],-np.inf))),
        bay_ever=int(np.any((np.abs(q[:,:,0])<=.2)&(q[:,:,1]>.2))))

def main():
    tasks=json.loads((HARD/'confirm_q16_tasks.json').read_text())['tasks']
    index={(t['state']['uid'],t['eta_index'],t['seed'],t['chain']):t for t in tasks if t['seed']==0}
    rows=[]
    for sc,name,old,new in [('ring_exchange','ring_gain_analysis.csv','TT','TF'),
                            ('toy_give_way','toy_loss_analysis.csv','FT','FF')]:
        for r in readcsv(OLD/name):
            uid=r['uid'];j=int(r['eta_index']);base=index[(uid,j,0,old)]
            p0=np.asarray(base['state']['physical']['positions']);g=np.asarray(base['state']['physical']['goals'])
            dd={}
            for ch in (old,new):
                t=index[(uid,j,0,ch)]
                with np.load(HARD/'results'/f"{t['id']}.npz") as z:
                    dd[ch]=z['positions'].copy()
            out=dict(scene=sc,uid=uid,eta_index=j,split=r['split'],b15_outcome=r['b15_outcome'],
                     seed=0,old=old,new=new,old_trajectory_steps=len(dd[old])-1,
                     new_trajectory_steps=len(dd[new])-1)
            if sc=='ring_exchange':
                gap=wrap(np.arctan2(g[:,1],g[:,0])-np.arctan2(p0[:,1],p0[:,0]))
                sign=np.sign(gap)
                out['ambiguous_agents']=int(np.sum(np.pi-np.abs(gap)<.05))
                for H in (10,20,80):
                    n0,l0=ring(dd[old],H);n1,l1=ring(dd[new],H)
                    out[f'H{H}_shortest_signed_net_delta']=float(np.mean(sign*(n1-n0)))
                    out[f'H{H}_ccw_net_delta']=float(np.mean(n1-n0))
                    out[f'H{H}_absolute_travel_delta']=float(np.sum(l1-l0))
                    out[f'H{H}_minimum_shortest_signed_net_old']=float(np.min(sign*n0))
                    out[f'H{H}_minimum_shortest_signed_net_new']=float(np.min(sign*n1))
            else:
                for H in (10,20,80):
                    m0=toy(dd[old],H);m1=toy(dd[new],H)
                    for k in m0:
                        out[f'H{H}_{k}_delta']=m1[k]-m0[k]
                    out[f'H{H}_worst_goal_progress_delta']=float(r[f'H{H}_worst_progress_delta']) if r[f'H{H}_worst_progress_delta'] else None
                out['FT_first_candidate']=r['FT_first_candidate']
                out['FF_first_candidate']=r['FF_first_candidate']
                out['FT_first_deadlock']=r['FT_first_deadlock']
                out['FF_first_deadlock']=r['FF_first_deadlock']
            rows.append(out)
    writecsv(HERE/'trajectory_geometry.csv',rows)
    print('seed0 paired trajectories',len(rows))

if __name__=='__main__':main()
