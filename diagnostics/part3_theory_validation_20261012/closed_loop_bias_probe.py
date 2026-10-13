"""Two-step fixed-key Toy tangent recurrence; diagnostic state propagation only.

No env.step, evaluator, or binary rollout labels. The plant update is the
audited p+=p+0.05u, v+=u map. K and H at the second state are independently
center-differenced; the two-step response is differenced as a separate call.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from h_bias_probe import HARD, biased, key_for, make_runtime, action

HERE=Path(__file__).resolve().parent

def main():
    import jax
    jax.config.update('jax_enable_x64',True)
    st=next(s for s in json.loads((HARD/'confirmation_manifest.json').read_text())['states'] if s['uid']=='hard_confirm_toy_1_820013')
    root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    eta=np.asarray(json.loads((HARD/'screening_manifest.json').read_text())['eta'][4],float)
    rt=make_runtime('toy_give_way');n=rt.make_env(st).positions.size
    direction=np.asarray(json.loads((HERE/'h_bias_results.json').read_text())['direction'])
    keys=[key_for(st['uid'],0,k,root) for k in range(2)]
    def set_state(env,x):
        env.positions=x[:n].reshape(-1,2).copy()
        env.velocities=x[n:].reshape(-1,2).copy()
    def do_action(x,theta,ch,key):
        env=rt.make_env(st);set_state(env,x)
        return action(biased(rt,theta,direction),env,key,eta,ch)[0].ravel()
    def next_state(x,u):return np.r_[x[:n]+rt.config.dt*u,u]
    env=rt.make_env(st);x0=np.r_[env.positions.ravel(),env.velocities.ravel()]
    rows=[];e=3e-4;ex=1e-4
    for ch in ('TT','TF','FT','FF'):
        u0=do_action(x0,0,ch,keys[0]);x1=next_state(x0,u0)
        h0=(do_action(x0,e,ch,keys[0])-do_action(x0,-e,ch,keys[0]))/(2*e)
        h1=(do_action(x1,e,ch,keys[1])-do_action(x1,-e,ch,keys[1]))/(2*e)
        k=[]
        for j in range(len(x1)):
            d=np.eye(len(x1))[j]*ex
            k.append((do_action(x1+d,0,ch,keys[1])-do_action(x1-d,0,ch,keys[1]))/(2*ex))
        k=np.stack(k,axis=1)
        acl=np.block([[np.eye(n)+rt.config.dt*k[:,:n],rt.config.dt*k[:,n:]],
                       [k[:,:n],k[:,n:]]])
        forcing=lambda h:np.r_[rt.config.dt*h,h]
        predicted=acl@forcing(h0)+forcing(h1)
        def two(theta):
            u=do_action(x0,theta,ch,keys[0]);xx=next_state(x0,u)
            return next_state(xx,do_action(xx,theta,ch,keys[1]))
        measured=(two(e)-two(-e))/(2*e)
        rows.append(dict(placement=ch,epsilon=e,k_state_epsilon=ex,
                         abs_error=float(np.linalg.norm(predicted-measured)),
                         rel_error=float(np.linalg.norm(predicted-measured)/max(np.linalg.norm(measured),1e-12)),
                         predicted_norm=float(np.linalg.norm(predicted)),measured_norm=float(np.linalg.norm(measured))))
    out=dict(state_uid=st['uid'],same_key_across_state_perturbations=True,
             horizon_steps=2,method='K,H independently differenced at x1; separate end-to-end two-step difference',
             rows=rows,new_rollouts=0)
    (HERE/'closed_loop_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
