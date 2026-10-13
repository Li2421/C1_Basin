"""Fixed-key closed-loop derivative recurrence at cached Ring timesteps.

No env.step is called. The controller action is reevaluated at perturbed
physical p/v around previously saved states; monitor state is irrelevant to
the Ring action graph before a terminal event.
"""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
for p in (str(ROOT),str(FIELD),str(FIELD/'source'),str(FIELD/'source/scripts')):sys.path.insert(0,p)
import numpy as np
from validation_stage2.run_rollouts import make_runtime
from validation_stage2.factorial_controller import action
from closed_loop_probe import key_for,HARD


def main():
    states=json.loads((HERE/'closed_loop_protocol.json').read_text())['selected_states']
    root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    arrays=np.load(HERE/'trajectory_sensitivity_arrays.npz')
    rt=make_runtime('ring_exchange')
    rows=[]
    for st in [s for s in states if s['scenario']=='ring_exchange']:
        uid=st['uid']
        for chain in ('TT','FF'):
            with np.load(HERE/'results_h80'/f'{uid}_{chain}_zero_center.npz') as z:
                pbase=z['positions'].copy();vbase=z['velocities'].copy();ubase=z['actions'].copy()
            sp=arrays[f'{uid}_{chain}_zero_0.003_position'];sv=arrays[f'{uid}_{chain}_zero_0.003_velocity']
            for t in (0,10,20,40,60):
                key=key_for(uid,0,t,root)
                def eval_at(x,eta):
                    env=rt.make_env(st)
                    env.positions=np.asarray(x[:8]).reshape(4,2).copy()
                    env.velocities=np.asarray(x[8:]).reshape(4,2).copy()
                    env.step_count=t
                    return action(rt,env,key,np.asarray(eta),chain)[0].ravel()
                x=np.r_[pbase[t].ravel(),vbase[t].ravel()]
                nominal=eval_at(x,[0.,0.,0.]);replay=float(np.linalg.norm(nominal-ubase[t].ravel()))
                dx=.0001;de=.003
                xu=[];ge=[]
                for j in range(len(x)):
                    e=np.zeros_like(x);e[j]=dx
                    xu.append((eval_at(x+e,[0.,0.,0.])-eval_at(x-e,[0.,0.,0.]))/(2*dx))
                for j in range(3):
                    e=np.zeros(3);e[j]=de
                    ge.append((eval_at(x,e)-eval_at(x,-e))/(2*de))
                au=np.stack(xu,axis=1);g=np.stack(ge,axis=1)
                s=np.r_[sp[t].reshape(8,3),sv[t].reshape(8,3)]
                pred_u=au@s+g
                obs=[]
                for j in range(3):
                    with (np.load(HERE/'results_h80'/f'{uid}_{chain}_zero_e{j}_+1_0.003.npz') as a,
                          np.load(HERE/'results_h80'/f'{uid}_{chain}_zero_e{j}_-1_0.003.npz') as b):
                        obs.append((a['actions'][t].ravel()-b['actions'][t].ravel())/(2*de))
                obs_u=np.stack(obs,axis=1)
                pred_next=np.r_[sp[t].reshape(8,3)+rt.config.dt*pred_u,pred_u]
                obs_next=np.r_[sp[t+1].reshape(8,3),sv[t+1].reshape(8,3)]
                rows.append(dict(uid=uid,chain=chain,t=t,replay_error=replay,
                    G_singular_values=np.linalg.svd(g,compute_uv=False).tolist(),
                    observed_action_sensitivity_norm=float(np.linalg.norm(obs_u)),
                    predicted_action_sensitivity_norm=float(np.linalg.norm(pred_u)),
                    action_absolute_error=float(np.linalg.norm(pred_u-obs_u)),
                    action_relative_error=float(np.linalg.norm(pred_u-obs_u)/max(np.linalg.norm(obs_u),1e-12)),
                    next_state_absolute_error=float(np.linalg.norm(pred_next-obs_next)),
                    next_state_relative_error=float(np.linalg.norm(pred_next-obs_next)/max(np.linalg.norm(obs_next),1e-12))))
    (HERE/'recurrence_spot_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
