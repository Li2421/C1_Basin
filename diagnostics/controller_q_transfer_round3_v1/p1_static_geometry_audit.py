"""Outcome-blind distances in the frozen P1 effective action representation."""
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    p=json.loads((HERE/'P1_PROTOCOL.json').read_text())
    names=p['controller_order']
    R3=[];R4=[]
    for name in names:
        with np.load(HERE/'p1_features'/f'{name}.npz') as z:
            R3.append(z['R3']);R4.append(z['R4_action'])
    R3=np.stack(R3);R4=np.stack(R4)
    assert R3.shape==(14,24,256) and R4.shape==(14,24,64)
    assert np.isfinite(R3).all() and np.isfinite(R4).all()
    base0=R4[:,:,:8];jac0=R4[:,:,8:32]
    baseprobe=R4[:,:,32:40];jacprobe=R4[:,:,40:64]
    def distances(x):
        return np.linalg.norm(x[:,None]-x[None],axis=-1)
    tri=np.triu_indices(14,1)
    result=dict(status='COMPLETED OUTCOME-BLIND STATIC ANALYSIS',controllers=names,states=24,
                x64=False,physical_steps=0,
                mean_pairwise_r3_distance=float(distances(R3).mean(-1)[tri].mean()),
                mean_pairwise_eta0_projected_action_distance=float(distances(base0).mean(-1)[tri].mean()),
                mean_pairwise_projected_jacobian_distance=float(distances(jac0).mean(-1)[tri].mean()),
                mean_t0_to_probe_action_change=float(np.linalg.norm(baseprobe-base0,axis=-1).mean()),
                mean_t0_to_probe_jacobian_change=float(np.linalg.norm(jacprobe-jac0,axis=-1).mean()),
                pairwise_eta0_distance=distances(base0).mean(-1).tolist(),
                pairwise_projected_jacobian_distance=distances(jac0).mean(-1).tolist(),
                interpretation='These are execution-response differences without labels; they neither prove controller-dependent eta ranking nor closed-loop transfer.')
    (HERE/'P1_STATIC_GEOMETRY_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:result[k] for k in ('mean_pairwise_eta0_projected_action_distance','mean_pairwise_projected_jacobian_distance','mean_t0_to_probe_action_change','mean_t0_to_probe_jacobian_change')})


if __name__=='__main__':main()
