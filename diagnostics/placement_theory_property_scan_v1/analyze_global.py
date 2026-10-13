"""Reuse bounded QMC volume and coarse voxel geometry; no new topology run."""
import json
from collections import deque
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
STRONG=HERE.parent/'placement_strong_hypotheses_v1'

def neighbors(z,n):
    for axis in range(3):
        for sign in (-1,1):
            x=list(z);x[axis]+=sign
            if 0<=x[axis]<n:yield tuple(x)

def components(cubes):
    unseen=set(cubes);count=0
    while unseen:
        count+=1;seed=unseen.pop();q=deque([seed])
        while q:
            for y in neighbors(q.popleft(),10000):
                if y in unseen:unseen.remove(y);q.append(y)
    return count

def voxel_geom(success,span):
    n=success.shape[0];cubes={tuple(x) for x in np.argwhere(success)}
    spacing=np.asarray(span)/(n-1)
    areas=[spacing[(axis+1)%3]*spacing[(axis+2)%3] for axis in range(3)]
    interior=exterior=0.
    for c in cubes:
        for axis in range(3):
            for sign in (-1,1):
                x=list(c);x[axis]+=sign
                if x[axis]<0 or x[axis]>=n:exterior+=areas[axis]
                elif tuple(x) not in cubes:interior+=areas[axis]
    base=components(cubes)
    articulation=0
    for c in cubes:
        if len(cubes)>1 and components(cubes-{c})>base:articulation+=1
    return dict(voxels=len(cubes),components_6=base,interior_face_area_proxy=float(interior),
        box_edge_face_area_proxy=float(exterior),articulation_voxels_6=articulation)

def main():
    old=json.loads((STRONG/'panel_analysis.json').read_text())
    new=json.loads((STRONG/'sobol128_analysis.json').read_text())
    rows=[]
    for r in new['rows']:
        a=next(x for x in old['states'] if x['uid']==r['uid'])
        gain=(64*a['gain_fraction']+128*r['independent_gain'])/192
        loss=(64*a['loss_fraction']+128*r['independent_loss'])/192
        rows.append(dict(uid=r['uid'],scene=r['scene'],gain_fraction=gain,loss_fraction=loss,
            net_fraction=gain-loss,observed_window=old['box']))
    z=np.load(STRONG/'grid9_rho_arrays.npz')
    span=np.asarray(old['box']['upper'])-np.asarray(old['box']['lower'])
    geom=[]
    for uid in ('hard_confirm_ring_1_920001','hard_confirm_toy_1_820001'):
        for n in (3,5,9):
            step=8//(n-1)
            for chain in ('TT','FF'):
                a=z[f'{uid}_{chain}'][::step,::step,::step]
                gg=voxel_geom(a>0,span)
                geom.append(dict(uid=uid,chain=chain,resolution=n,**gg))
    stats={}
    for scene in ('ring_exchange','toy_give_way','both'):
        rr=[r for r in rows if scene=='both' or r['scene']==scene]
        stats[scene]=dict(states=len(rr),mean_gain_fraction=float(np.mean([r['gain_fraction'] for r in rr])),
            mean_loss_fraction=float(np.mean([r['loss_fraction'] for r in rr])),
            mean_net_fraction=float(np.mean([r['net_fraction'] for r in rr])),
            states_with_gain=sum(r['gain_fraction']>0 for r in rr),
            states_with_loss=sum(r['loss_fraction']>0 for r in rr))
    out=dict(volume=stats,voxel_geometry=geom,
        caveat='Voxel surface area and articulation counts are resolution-sensitive occupancy proxies, not certified basin invariants. Failure voxels are binary evaluator failures.')
    (HERE/'global_geometry_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/'global_geometry_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
