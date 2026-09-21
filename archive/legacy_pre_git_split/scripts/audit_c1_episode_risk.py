"""Replay risk definitions on existing full traces, without changing policies.

Safety is interpreted as the zero-residual C1 controller: its task candidate
is its projected control, not the unprojected nominal action. For C1, use the
actual pre-final-projection candidate stored by the evaluator.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.cbf import CBFConfig
from single_integrator.c1.risk.evaluation import audit_trace
from single_integrator.c1.risk.risk_v1 import RiskV1Config,trajectory_risk_v1
from single_integrator.c1.risk.risk_v2 import RiskV2Config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--arm',choices=('safety','c1'),required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    jax.config.update('jax_enable_x64',True)
    contract=json.loads(args.contract.read_text())
    plant=Config(**contract['environment']);cbf=CBFConfig(**contract['cbf'])
    # Common, explicitly recorded diagnostic parameters, not a claim these
    # historical policies were trained using v2.
    risk=RiskV2Config(kappa=2.,goal_tolerance=plant.goal_tolerance)
    old=RiskV1Config(kappa=2.,goal_tolerance=plant.goal_tolerance)
    rows=[]
    for summary in json.loads((args.folder/'summary.json').read_text())['rollouts']:
        rid=summary['rollout_id']
        with np.load(args.folder/f'rollout_{rid:04d}.npz') as data:
            trace={k:np.asarray(data[k]) for k in data.files}
        applied=trace['executed_velocity'].reshape(-1,4)
        if args.arm=='safety':
            trace.update(c1_safe=applied,c1_candidate=applied,c1_correction=np.zeros_like(applied))
        else:
            candidate=trace.get('c1_candidate',trace['raw_policy_velocity']).reshape(-1,4)
            # Earlier C1 traces do not store u_safe. Do not fabricate correction
            # metrics; this audit only reports risk, which doesn't need u_safe.
            trace.update(c1_candidate=candidate,c1_safe=applied,c1_correction=np.zeros_like(applied))
        measured,fields=audit_trace(trace,plant,cbf,risk)
        positions=np.concatenate((trace['positions_before'][:1],trace['positions']),axis=0)
        legacy=None
        if len(applied)>old.window_steps(plant.dt):
            legacy=float(trajectory_risk_v1(fields['c1_cone_risk_t'],positions,
                    GiveWayEnv(plant).goals,old,old.window_steps(plant.dt))['trajectory_risk'][0])
        row=dict(rollout_id=rid,outcome=summary['outcome'],steps=len(applied),
                 v1_1_risk=legacy,v2_risk=measured['trajectory_risk'],
                 exposure_risk=measured['exposure_risk'],terminal_risk=measured['terminal_risk'],
                 censored=measured['censored'])
        rows.append(row);print(row,flush=True)
    def aggregate(name,outcome):
        values=[r[name] for r in rows if (r['outcome']=='success')==outcome and r[name] is not None]
        return dict(n=len(values),mean=float(np.mean(values)) if values else None,
                    min=min(values,default=None),max=max(values,default=None))
    aggregates={name:{'success':aggregate(name,True),'failure':aggregate(name,False)}
                for name in ('v1_1_risk','v2_risk')}
    args.out.write_text(json.dumps(dict(arm=args.arm,risk=asdict(risk),
        interpretation='post-hoc diagnostic, not v2-trained results; no threshold tuned to these outcomes',
        task_candidate='projected Safety control' if args.arm=='safety' else 'actual C1 candidate',
        aggregate=aggregates,rollouts=rows),indent=2)+'\n')


if __name__=='__main__':main()
