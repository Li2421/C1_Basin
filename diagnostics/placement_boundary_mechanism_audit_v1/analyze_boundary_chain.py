"""Align action, trajectory and margin derivatives at each measured Ring boundary."""
import json
import sys
import numpy as np
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity


def load(uid,chain,eta):
    eid=eta_identity(eta)[0]
    with np.load(HERE/'boundary_normal_results'/f'{uid}_{chain}_{eid}.npz') as z:
        return {k:z[k].copy() for k in z.files}


def main():
    rows=[];centers=json.loads((HERE/'boundary_normal_protocol.json').read_text())
    normal={ (r['uid'],r['chain']):r for r in json.loads((HERE/'boundary_normals.json').read_text())}
    for c in centers:
        uid,chain=c['uid'],c['chain'];r=normal[(uid,chain)];eta=np.asarray(c['eta'])
        measures={}
        for eps in (.002,.0005):
            gs=[];ss=[]
            for j in range(3):
                m=eta.copy();m[j]-=eps;p=eta.copy();p[j]+=eps
                zm,zp=load(uid,chain,m),load(uid,chain,p)
                gs.append((zp['actions'][0].ravel()-zm['actions'][0].ravel())/(2*eps))
                ss.append((zp['positions'][80].ravel()-zm['positions'][80].ravel())/(2*eps))
            g=np.stack(gs,axis=1);s=np.stack(ss,axis=1)
            measures[eps]=(g,s)
        g,s=measures[.0005];g2,s2=measures[.002]
        rows.append(dict(uid=uid,chain=chain,boundary_fraction=c['fraction'],
            G_singular_values=np.linalg.svd(g,compute_uv=False).tolist(),
            G_eps_stability=float(np.linalg.norm(g-g2)/max(np.linalg.norm(g),1e-12)),
            S80_singular_values=np.linalg.svd(s,compute_uv=False).tolist(),
            S80_eps_stability=float(np.linalg.norm(s-s2)/max(np.linalg.norm(s),1e-12)),
            G=g.tolist(),S80=s.tolist(),rho_gradient=r['gradient_eps0005'],
            rho_gradient_stability=r['gradient_stability']))
    (HERE/'boundary_mechanism_chain.json').write_text(json.dumps(rows,indent=2)+'\n')
    paired=[]
    for uid in sorted({r['uid'] for r in rows}):
        a,b=[r for r in rows if r['uid']==uid]
        for key in ('G','S80'):
            x,y=np.asarray(a[key]),np.asarray(b[key])
            paired.append(dict(uid=uid,object=key,relative_difference=float(np.linalg.norm(y-x)/max(np.linalg.norm(y),1e-12))))
    (HERE/'boundary_mechanism_pair_differences.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(json.dumps(paired,indent=2))


if __name__=='__main__':main()
