"""Direct finite-eta derivative of B_t eta on saved paired Ring trajectories."""
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
HERE=Path(__file__).resolve().parent
STRONG=HERE.parent/'placement_strong_hypotheses_v1'
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
sys.path.insert(0,str(HERE.parent/'placement_boundary_mechanism_audit_v1'))
from closed_loop_probe import key_for
from validation_stage2.run_rollouts import make_runtime

def trace(uid,chain,kind,axis=None,sign=None,epsilon=.0005):
    a='None' if axis is None else str(axis);s='None' if sign is None else str(sign)
    name=f'{uid}_root{chain}_eval{chain}_{kind}_e{0.0 if kind=="center" else epsilon}_a{a}_s{s}.npz'
    with np.load(STRONG/'h1_gradient_results'/name) as z:
        return z['positions'].copy(),z['velocities'].copy(),z['noise_keys'].copy()

def compute_B(rt,st,positions,velocities,t,key):
    env=rt.make_env(st)
    env.positions=np.asarray(positions[t],float).copy()
    env.velocities=np.asarray(velocities[t],float).copy()
    env.step_count=t
    obs,safety,flow,safe,_,_=rt.prepare(env,key,np.zeros(3))
    b=np.stack(rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values,axis=-1).reshape(-1,3)
    active=np.r_[safety.A@safe.ravel()-safety.lower<1e-5,
                 rt.config.max_speed-np.linalg.norm(safe,axis=1)<1e-5]
    return b,hashlib.sha256(active.astype('uint8').tobytes()).hexdigest()[:16]

def main():
    centers=json.loads((STRONG/'h1_gradient_protocol.json').read_text())['centers']
    states={s['uid']:s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states']}
    rt=make_runtime('ring_exchange');rows=[];raw={}
    for center in centers:
        uid=center['uid'];chain=center['root_chain'];eta=np.asarray(center['eta'],float)
        cp,cv,ck=trace(uid,chain,'center')
        for t in (0,100,200,300):
            key=key_for(uid,0,t,2026100601)
            B0,base_signature=compute_B(rt,states[uid],cp,cv,t,key)
            for eps in (.0005,.002):
                side={(axis,sign):trace(uid,chain,'fd',axis,sign,eps) for axis in range(3) for sign in (-1,1)}
                if any(len(side[k][0])<=t for k in side):continue
                measured=[];direct=[];indirect=[];switches=[]
                for axis in range(3):
                    bp,sp=compute_B(rt,states[uid],*side[(axis,1)][:2],t,key)
                    bm,sm=compute_B(rt,states[uid],*side[(axis,-1)][:2],t,key)
                    ep=eta.copy();em=eta.copy();ep[axis]+=eps;em[axis]-=eps
                    total=(bp@ep-bm@em)/(2*eps)
                    d=B0[:,axis]
                    i=((bp-bm)/(2*eps))@eta
                    measured.append(total);direct.append(d);indirect.append(i)
                    switches.append(sp!=sm or sp!=base_signature)
                measured=np.stack(measured,axis=1);direct=np.stack(direct,axis=1);indirect=np.stack(indirect,axis=1)
                err=float(np.linalg.norm(measured-direct-indirect)/max(np.linalg.norm(measured),1e-12))
                row=dict(uid=uid,chain=chain,physical_step=t,epsilon=eps,eta=eta.tolist(),
                    direct_norm=float(np.linalg.norm(direct)),indirect_norm=float(np.linalg.norm(indirect)),
                    measured_total_norm=float(np.linalg.norm(measured)),
                    indirect_to_direct_ratio=float(np.linalg.norm(indirect)/max(np.linalg.norm(direct),1e-12)),
                    indirect_to_total_ratio=float(np.linalg.norm(indirect)/max(np.linalg.norm(measured),1e-12)),
                    decomposition_relative_error=err,
                    basis_projection_signature_switches=switches,
                    correction_norm=float(np.linalg.norm(B0@eta)))
                rows.append(row)
                tag=f'{uid}_{chain}_t{t}_e{eps}'
                raw[tag+'_B0']=B0;raw[tag+'_d_direct']=direct;raw[tag+'_d_indirect']=indirect;raw[tag+'_d_total']=measured
    scale=[]
    for r in rows:
        if r['epsilon']!=.0005:continue
        other=next(q for q in rows if q['uid']==r['uid'] and q['chain']==r['chain'] and
                   q['physical_step']==r['physical_step'] and q['epsilon']==.002)
        a=raw[f"{r['uid']}_{r['chain']}_t{r['physical_step']}_e0.0005_d_indirect"]
        b=raw[f"{r['uid']}_{r['chain']}_t{r['physical_step']}_e0.002_d_indirect"]
        rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))
        r['indirect_scale_change']=rel;other['indirect_scale_change']=rel;scale.append(rel)
    fine=[r for r in rows if r['epsilon']==.0005]
    out=dict(points=len(rows),max_decomposition_error=max(r['decomposition_relative_error'] for r in rows),
        median_indirect_to_direct_at_t0=float(np.median([r['indirect_to_direct_ratio'] for r in fine if r['physical_step']==0])),
        median_indirect_to_direct_late=float(np.median([r['indirect_to_direct_ratio'] for r in fine if r['physical_step']>0])),
        late_ratio_quantiles=np.quantile([r['indirect_to_direct_ratio'] for r in fine if r['physical_step']>0],[0,.25,.5,.75,1]).tolist(),
        late_scale_change_quantiles=np.quantile([r['indirect_scale_change'] for r in fine if r['physical_step']>0],[0,.25,.5,.75,1]).tolist(),
        branch_switch_rows=sum(any(r['basis_projection_signature_switches']) for r in rows),
        note='Identity is product rule at finite eta using true paired trajectories; branch switches may make the finite-difference derivative only a secant.')
    np.savez_compressed(HERE/'basis_derivative_raw.npz',**raw)
    (HERE/'basis_derivative_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/'basis_derivative_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
