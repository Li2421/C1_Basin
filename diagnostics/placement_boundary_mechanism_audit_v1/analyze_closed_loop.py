"""Analyze paired trajectory and evaluator-aligned diagnostic margin data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PROTO=json.loads((HERE/'closed_loop_protocol.json').read_text())


def stem(uid,p,center,variant):
    return f'{uid}_{p}_{center}_{variant}'


def load(folder,uid,p,center,variant):
    name=stem(uid,p,center,variant)
    d=json.loads((folder/(name+'.json')).read_text())
    with np.load(folder/(name+'.npz')) as z:
        a={k:z[k].copy() for k in z.files}
    return d,a


def snorm(x):return float(np.linalg.norm(x))
def cosine(x,y):return float(np.sum(x*y)/max(snorm(x)*snorm(y),1e-12))
def save(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def run():
    rows=[];margin_rows=[];pred=[];arrays={}
    for state in PROTO['selected_states']:
        uid=state['uid'];scene=state['scenario']
        for center,folder in [('zero',HERE/'results_h80'),('candidate4',HERE/'results_full')]:
            for p in ('TT','FF'):
                base,zb=load(folder,uid,p,center,'center')
                traj={};grad={};signatures={}
                for eps in (.01,.003):
                    sp=[];sv=[];g=[];sign=[];valid=[]
                    for j in range(3):
                        minus,zm=load(folder,uid,p,center,f'e{j}_-1_{eps}')
                        plus,zp=load(folder,uid,p,center,f'e{j}_+1_{eps}')
                        n=min(len(zm['positions']),len(zp['positions']),len(zb['positions']))
                        sp.append((zp['positions'][:n]-zm['positions'][:n])/(2*eps))
                        sv.append((zp['velocities'][:n]-zm['velocities'][:n])/(2*eps))
                        g.append((plus['rho']-minus['rho'])/(2*eps) if plus['rho'] is not None and minus['rho'] is not None else float('nan'))
                        sign.append(dict(plus_termination=plus['termination'],minus_termination=minus['termination'],
                            plus_goal_active=plus['goal_active_step'],minus_goal_active=minus['goal_active_step'],
                            plus_safety_active=plus['safety_active_step'],minus_safety_active=minus['safety_active_step'],
                            plus_margin_type='goal' if plus['goal_ever_margin']<=plus['safety_always_margin'] else 'safety',
                            minus_margin_type='goal' if minus['goal_ever_margin']<=minus['safety_always_margin'] else 'safety'))
                        valid.append(n)
                    n=min(valid)
                    pos=np.stack([v[:n] for v in sp],axis=-1)
                    vel=np.stack([v[:n] for v in sv],axis=-1)
                    traj[eps]=(pos,vel)
                    grad[eps]=np.asarray(g)
                    signatures[eps]=sign
                    rel=pos[:,0]-pos[:,1] if pos.shape[1]==2 else np.stack([pos[:,i]-pos[:,j] for i in range(pos.shape[1]) for j in range(i+1,pos.shape[1])],axis=1)
                    goal_error=np.linalg.norm(state['physical']['goals']-zb['positions'][:n],axis=2)
                    sampled=sorted(set([0,1,10,20,40,80,120,160,200,n-1]) & set(range(n)))
                    rows.append(dict(scene=scene,uid=uid,placement=p,center=center,epsilon=eps,common_prefix=n-1,
                        sample_steps=sampled,position_sensitivity_norm=[snorm(pos[t]) for t in sampled],
                        velocity_sensitivity_norm=[snorm(vel[t]) for t in sampled],
                        relative_position_sensitivity_norm=[snorm(rel[t]) for t in sampled],
                        goal_error_at_sample_steps=[goal_error[t].tolist() for t in sampled],
                        singular_values_t80=np.linalg.svd(pos[min(80,n-1)].reshape(-1,3),compute_uv=False).tolist()))
                    arrays[f'{uid}_{p}_{center}_{eps}_position']=pos
                    arrays[f'{uid}_{p}_{center}_{eps}_velocity']=vel
                n=min(len(traj[.01][0]),len(traj[.003][0]))
                stability=[snorm(traj[.01][0][t]-traj[.003][0][t])/max(snorm(traj[.003][0][t]),1e-12) for t in (10,40,min(80,n-1))]
                margin_rows.append(dict(scene=scene,uid=uid,placement=p,center=center,baseline_term=base['termination'],
                    baseline_rho=base['rho'],baseline_success=base['success'],baseline_goal_active=base['goal_active_step'],
                    baseline_safety_active=base['safety_active_step'],
                    baseline_margin_type='goal' if base['goal_ever_margin']<=base['safety_always_margin'] else 'safety',
                    gradient_eps01=grad[.01].tolist(),gradient_eps003=grad[.003].tolist(),
                    gradient_stability=snorm(grad[.01]-grad[.003])/max(snorm(grad[.003]),1e-12),
                    trajectory_sensitivity_stability=stability,
                    side_signatures_eps01=signatures[.01],side_signatures_eps003=signatures[.003]))
                # Held-out diagonal perturbation predicts full trajectories and rho.
                # Full candidate4 has no independent diagonal run: this is short-horizon only.
                if center=='zero':
                    eps=.003; direction=np.array([1.,1.,1.])/np.sqrt(3)
                    pred.append(dict(uid=uid,placement=p,center=center,predicted_position_t80=(traj[eps][0][80].reshape(-1,3)@(eps*direction)).tolist(),
                        predicted_margin_change=float(grad[eps]@(eps*direction))))
    np.savez_compressed(HERE/'trajectory_sensitivity_arrays.npz',**arrays)
    save('trajectory_sensitivity.json',rows)
    save('rho_gradients.json',margin_rows)
    save('unvalidated_first_order_predictions.json',pred)
    paired=[]
    for st in PROTO['selected_states']:
        uid=st['uid']
        for center in ('zero','candidate4'):
            for eps in (.01,.003):
                a=arrays[f'{uid}_TT_{center}_{eps}_position'];b=arrays[f'{uid}_FF_{center}_{eps}_position']
                n=min(len(a),len(b));t=min(80,n-1)
                paired.append(dict(scene=st['scenario'],uid=uid,center=center,epsilon=eps,t=t,
                    relative_S_difference=snorm(a[t]-b[t])/max(snorm(b[t]),1e-12),
                    S_TT_norm=snorm(a[t]),S_FF_norm=snorm(b[t]),
                    principal_angles_deg=np.degrees(np.arccos(np.clip(np.linalg.svd(np.linalg.qr(a[t].reshape(-1,3))[0].T@
                        np.linalg.qr(b[t].reshape(-1,3))[0],compute_uv=False),0,1))).tolist()))
    save('paired_trajectory_sensitivity.json',paired)
    print(json.dumps({'trajectory_rows':len(rows),'margin_rows':len(margin_rows),'paired':paired},indent=2))


if __name__=='__main__':run()
