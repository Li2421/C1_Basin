"""Audit R_risk_v0_1 on paired MAC-only, Safety, and C1 rollout traces.

This is an evaluation calculation only.  It reuses the C1 V0.1 soft activity
and analytic cone routines without modifying any controller output.
"""
import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.c1.risk.soft_activity import SoftRiskConfig, soft_risk_diagnostics
from single_integrator.c1.risk.risk_function import trajectory_risk
from single_integrator.environment import Config, GiveWayEnv


def arm_rows(folder, summaries, walls, plant, cbf, risk, task_input='nominal'):
    score = jax.jit(lambda f, blocks, h, s: soft_risk_diagnostics(f, blocks, h, s, risk))
    rows = []
    for summary in summaries:
        rid = int(summary['rollout_id'])
        with np.load(folder / f'rollout_{rid:04d}.npz') as trace:
            risks, hard, activity, margins, residuals = [], [], [], [], []
            task=trace['u_safe'] if task_input=='projected' else trace['u_nom']
            for nominal, applied, position in zip(task, trace['u_safe'], trace['positions_before']):
                A, b, geometry = barrier_constraints(dict(positions=position, walls=walls, config=plant.to_dict()), cbf)
                h = np.r_[geometry['pairwise_h'], geometry['wall_h'].reshape(-1)]
                residual = A @ applied.reshape(4) - b
                d = score(nominal, A.reshape(17, 2, 2), h, residual)
                risks.append(float(d['risk']))
                hard.append(bool(np.any(np.asarray(d['active_mask']))))
                activity.append(float(np.max(np.asarray(d['local_activity']))))
                margins.append(np.asarray(d['margins']))
                residuals.append(residual)
        values = np.asarray(risks)
        rows.append(dict(rollout_id=rid, outcome=summary['outcome'], episode_steps=len(values),
                         R_risk=float(trajectory_risk(values, risk)), mean_risk=float(values.mean()),
                         max_risk=float(values.max()), hard_active_fraction=float(np.mean(hard)),
                         mean_local_activity=float(np.mean(activity)),
                         min_cbf_residual=float(np.min(residuals))))
    return rows


def aggregate(rows):
    return dict(n_rollouts=len(rows), mean_R_risk=float(np.mean([x['R_risk'] for x in rows])),
                median_R_risk=float(np.median([x['R_risk'] for x in rows])),
                mean_risk=float(np.mean([x['mean_risk'] for x in rows])),
                mean_hard_active_fraction=float(np.mean([x['hard_active_fraction'] for x in rows])),
                mean_local_activity=float(np.mean([x['mean_local_activity'] for x in rows])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, required=True)
    p.add_argument('--c1-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--historical-nominal-task',action='store_true',
                   help='Replay the historical nominal-task Safety diagnostic, not the zero-residual C1 objective')
    args = p.parse_args()
    contract = json.loads((args.baseline_dir / 'config.json').read_text())
    c1_contract = json.loads((args.c1_dir / 'config.json').read_text())
    plant, cbf = Config(**contract['environment']), CBFConfig(**contract['cbf'])
    if c1_contract['environment'] != plant.to_dict() or c1_contract['seed'] != contract['seed']:
        raise ValueError('C1 and baseline evaluation contracts differ')
    risk = SoftRiskConfig(rho=.05, active_tol=1e-7, D0=.1, kappa=2., alpha=.5, p=4.,
                          tau_h=.01, tau_s=.01, candidate_sigmas=6.)
    walls = GiveWayEnv(plant).walls
    inputs = dict(mac_only=(args.baseline_dir / 'mac_only', json.loads((args.baseline_dir/'mac_only/summary.json').read_text())['rollouts']),
                  safety=(args.baseline_dir / 'mac_cbf', json.loads((args.baseline_dir/'mac_cbf/summary.json').read_text())['rollouts']),
                  c1=(args.c1_dir, json.loads((args.c1_dir/'summary.json').read_text())['rollouts']))
    arms = {name: arm_rows(folder, summaries, walls, plant, cbf, risk,
                          task_input='projected' if name=='safety' and not args.historical_nominal_task else 'nominal')
            for name, (folder, summaries) in inputs.items()}
    if not all([x['rollout_id'] for x in arms['mac_only']] == [x['rollout_id'] for x in arms[name]] for name in ('safety', 'c1')):
        raise ValueError('unpaired rollout ids')
    paired = {name: dict(mean_delta_R_risk=float(np.mean([c['R_risk']-b['R_risk'] for b, c in zip(arms['safety'], arms[name])])),
                         improved_count=sum(c['R_risk'] < b['R_risk'] for b, c in zip(arms['safety'], arms[name])),
                         worsened_count=sum(c['R_risk'] > b['R_risk'] for b, c in zip(arms['safety'], arms[name])))
              for name in ('mac_only', 'c1')}
    args.out.write_text(json.dumps(dict(risk_version='R_risk_v0_1', risk=risk.__dict__,
                                         task_protocol='historical_nominal' if args.historical_nominal_task else 'c1_candidate_with_zero_residual_safety',
                                         semantics='MAC-only uses its executed nominal velocity in s=A u-b; Safety and C1 use their final projected velocity.',
                                         aggregate={name: aggregate(rows) for name, rows in arms.items()}, arms=arms,
                                         paired_minus_safety=paired), indent=2)+'\n')


if __name__ == '__main__':
    main()
