"""Recompute C1's provisional risk on matched MAC-only, Safety, and C1 traces.

The ``binding`` variant exactly matches the activity rule used by C1 training:
``h <= rho`` *and* the applied control is CBF-binding.  MAC-only has no CBF
projection, so a second, explicitly labelled ``lookahead`` diagnostic removes
only the binding requirement.  It is not the trained objective.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, GiveWayEnv


RHO, ACTIVE_TOL, D0, KAPPA, ALPHA, POWER = .05, 1e-7, .1, 8., .5, 4.


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def signed_margin(x, rays):
    """Use the authoritative R_risk_v0 geometry; no duplicate classifier."""
    import jax
    from single_integrator.c1.risk.risk_function import cone_geometry
    with jax.experimental.enable_x64():
        return float(cone_geometry(np.asarray(x), np.asarray(rays).reshape(-1, 2))[0])

def step_risk(nominal, applied, positions, walls, plant, cbf, variant):
    snapshot = dict(positions=positions, walls=walls, config=plant.to_dict())
    A, b, geometry = barrier_constraints(snapshot, cbf)
    h = np.r_[geometry['pairwise_h'], geometry['wall_h'].reshape(-1)]
    if variant == 'binding':
        active = (h <= RHO) & (np.abs(A @ applied.reshape(4) - b) <= ACTIVE_TOL)
    elif variant == 'lookahead':
        active = h <= RHO
    else:
        raise ValueError(variant)
    if not np.any(active):
        return 0., 0
    blocks = A.reshape(17, 2, 2)[active]
    values = []
    for agent in range(2):
        margin = signed_margin(-nominal[agent], blocks[:, agent])
        value = 1. if np.isneginf(margin) else 1. / (1. + np.exp(-KAPPA * (D0 - margin) / D0))
        values.append(value)
    return float(max(values)), int(active.sum())


def trajectory(trace, walls, plant, cbf, variant):
    nominal, applied, positions = trace['u_nom'], trace['u_safe'], trace['positions_before']
    risks, active_counts = zip(*(step_risk(n, u, p, walls, plant, cbf, variant)
                                 for n, u, p in zip(nominal, applied, positions)))
    risks = np.asarray(risks)
    value = ALPHA * np.mean(risks ** POWER) ** (1. / POWER) + (1. - ALPHA) * risks.mean()
    return dict(R_risk=float(value), mean_risk=float(risks.mean()), max_risk=float(risks.max()),
                active_step_fraction=float(np.count_nonzero(risks) / len(risks)),
                mean_active_constraints=float(np.mean(active_counts)), episode_steps=int(len(risks)))


def load_arm(folder, arm, walls, plant, cbf):
    summaries = json.loads((folder / arm / 'summary.json').read_text())['rollouts']
    rows = []
    for summary in summaries:
        rid = int(summary['rollout_id'])
        with np.load(folder / arm / f'rollout_{rid:04d}.npz') as trace:
            row = dict(rollout_id=rid, outcome=summary['outcome'],
                       binding=trajectory(trace, walls, plant, cbf, 'binding'),
                       lookahead=trajectory(trace, walls, plant, cbf, 'lookahead'))
        rows.append(row)
    return rows


def aggregate(rows, variant):
    values = np.asarray([row[variant]['R_risk'] for row in rows])
    return dict(mean_R_risk=float(values.mean()), median_R_risk=float(np.median(values)),
                mean_max_risk=float(np.mean([row[variant]['max_risk'] for row in rows])),
                mean_active_step_fraction=float(np.mean([row[variant]['active_step_fraction'] for row in rows])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--c1-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((args.baseline_dir / 'config.json').read_text())
    plant = Config(**contract['environment'])
    walls = GiveWayEnv(plant).walls
    cbf = CBFConfig(**contract['cbf'])
    c1_summary = json.loads((args.c1_dir / 'summary.json').read_text())['rollouts']
    c1_rows = []
    for summary in c1_summary:
        rid = int(summary['rollout_id'])
        with np.load(args.c1_dir / f'rollout_{rid:04d}.npz') as trace:
            c1_rows.append(dict(rollout_id=rid, outcome=summary['outcome'],
                                binding=trajectory(trace, walls, plant, cbf, 'binding'),
                                lookahead=trajectory(trace, walls, plant, cbf, 'lookahead')))
    arms = dict(mac_only=load_arm(args.baseline_dir, 'mac_only', walls, plant, cbf),
                safety=load_arm(args.baseline_dir, 'mac_cbf', walls, plant, cbf), c1=c1_rows)
    report = dict(risk_definition=dict(rho=RHO, active_tol=ACTIVE_TOL, D0=D0, kappa=KAPPA,
                                       alpha=ALPHA, p=POWER),
                  caveat='binding matches the trained activity rule; lookahead is a diagnostic.',
                  aggregate={arm: {variant: aggregate(rows, variant) for variant in ('binding', 'lookahead')}
                             for arm, rows in arms.items()}, arms=arms)
    safety, c1 = arms['safety'], arms['c1']
    report['paired_c1_minus_safety'] = {variant: dict(
        mean_delta_R_risk=float(np.mean([c[variant]['R_risk'] - s[variant]['R_risk'] for c, s in zip(c1, safety)])),
        improved_count=int(sum(c[variant]['R_risk'] < s[variant]['R_risk'] for c, s in zip(c1, safety))),
        worsened_count=int(sum(c[variant]['R_risk'] > s[variant]['R_risk'] for c, s in zip(c1, safety))),
        unchanged_count=int(sum(c[variant]['R_risk'] == s[variant]['R_risk'] for c, s in zip(c1, safety))))
        for variant in ('binding', 'lookahead')}
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
