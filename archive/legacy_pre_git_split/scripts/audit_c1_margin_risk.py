"""Audit where R_risk_v2 obtains useful or unstable cone-margin gradients.

This is a read-only replay of a saved C1 residual on fixed starts/noise.  It
reports coverage of the cone-risk sigmoid's 10--90% transition interval and
whether large risk changes coincide with discrete cone/candidate switches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.risk.evaluation import risk_from_metadata
from single_integrator.c1.socp import ExactProjection
from single_integrator.c1.train import rollout_terms
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config
from single_integrator.evaluate import load_policy


def fraction(mask):
    return float(np.mean(mask)) if mask.size else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True,
                        help='NPZ with initial_positions [N,2,2] and noise [N,T,4].')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=2,
                        help='Replay this prefix of fixed inputs; use 0 for all.')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'output already exists: {args.out}')
    jax.config.update('jax_enable_x64', True)

    saved = pickle.loads(args.residual.read_bytes())
    metadata = saved['metadata']
    if metadata.get('risk_version') != 'R_risk_v2':
        raise ValueError('this audit currently requires an R_risk_v2 residual')
    risk = risk_from_metadata(metadata)
    with np.load(args.inputs, allow_pickle=False) as data:
        starts = np.asarray(data['initial_positions'], dtype=np.float64)
        noise = np.asarray(data['noise'])
    count = len(starts) if args.episodes == 0 else args.episodes
    if (starts.ndim != 3 or starts.shape[1:] != (2, 2) or noise.ndim != 3
            or noise.shape[0] != len(starts) or noise.shape[2] != 4
            or not 1 <= count <= len(starts)):
        raise ValueError('invalid episode count or fixed inputs')
    starts, noise = starts[:count], noise[:count]

    baseline, provenance = load_policy(Path(metadata['baseline_checkpoint']))
    plant = Config(**provenance['evaluation_environment'])
    model = ResidualCorrection(hidden_dims=tuple(metadata['architecture']['hidden_dims']),
                               layer_norm=metadata['architecture']['layer_norm'])
    field = ResidualFlowField(baseline, model)
    field.baseline_sample = jax.jit(field.baseline_sample)
    field.correction = jax.jit(field.correction)
    _, _, trajectory, details = rollout_terms(
        jax.tree_util.tree_map(jnp.asarray, saved['params']), field, ExactProjection(),
        jnp.asarray(starts), jnp.asarray(noise), plant, CBFConfig(), risk, return_details=True)
    details = jax.device_get(details)

    margins = np.asarray(details['margins'])             # [B,T,2]
    cone_risk = np.asarray(details['cone_risk'])         # [B,T,2]
    step_risk = np.asarray(details['risk'])               # [B,T]
    valid = np.asarray(details['episode_mask'], bool)
    kinds = np.asarray(details['cone_types'])
    candidate = np.asarray(details['candidate_mask'], bool)
    activity = np.asarray(details['local_activity'])
    finite = np.isfinite(margins) & valid[..., None]

    # r = sigmoid(kappa * (1 - m / D0)); solve r=0.9 and r=0.1 for m.
    logit90 = np.log(.9 / .1)
    low = risk.D0 * (1 - logit90 / risk.kappa)
    high = risk.D0 * (1 + logit90 / risk.kappa)
    values = margins[finite]
    cone_values = cone_risk[finite]
    derivative = (risk.kappa / risk.D0) * cone_values * (1 - cone_values)
    transition = finite & (margins >= low) & (margins <= high)
    low_saturation = finite & (margins > high)
    high_saturation = finite & (margins < low)

    pair_valid = valid[:, 1:] & valid[:, :-1]
    kind_switch = (kinds[:, 1:] != kinds[:, :-1]) & pair_valid[..., None]
    candidate_switch = (candidate[:, 1:] != candidate[:, :-1]) & pair_valid[..., None]
    risk_delta = np.abs(step_risk[:, 1:] - step_risk[:, :-1])
    large = (risk_delta >= .2) & pair_valid
    any_kind_switch = np.any(kind_switch, axis=-1)
    any_candidate_switch = np.any(candidate_switch, axis=-1)
    cone_change = np.max(np.abs(cone_risk[:, 1:] - cone_risk[:, :-1]), axis=-1) >= .2
    activity_change = np.max(np.abs(activity[:, 1:] - activity[:, :-1]), axis=-1) >= .2
    was_transition = np.any(transition[:, 1:] | transition[:, :-1], axis=-1)
    common_finite = finite[:, 1:] & finite[:, :-1]
    margin_delta = np.abs(margins[:, 1:] - margins[:, :-1])
    finite_margin_delta = margin_delta[common_finite]
    # A change of this width can traverse the whole 10--90% sigmoid interval.
    whole_sigmoid_crossing = common_finite & (margin_delta >= high - low)

    episodes = []
    for index in range(count):
        mask = finite[index]
        episodes.append(dict(
            input_index=index,
            executed_steps=int(valid[index].sum()),
            trajectory_risk=float(np.asarray(trajectory)[index]),
            finite_margin_count=int(mask.sum()),
            margin_min=float(margins[index][mask].min()) if mask.any() else None,
            margin_max=float(margins[index][mask].max()) if mask.any() else None,
            transition_fraction=fraction(transition[index][mask]),
            large_risk_changes=int(large[index].sum()),
            large_change_with_cone_switch=int(np.sum(large[index] & any_kind_switch[index])),
            large_change_with_candidate_switch=int(np.sum(large[index] & any_candidate_switch[index])),
            large_change_with_cone_risk_change=int(np.sum(large[index] & cone_change[index])),
            large_change_with_activity_change=int(np.sum(large[index] & activity_change[index])),
            large_change_adjacent_to_transition_margin=int(np.sum(large[index] & was_transition[index])),
            whole_sigmoid_margin_crossings=int(whole_sigmoid_crossing[index].sum()),
            whole_crossing_with_cone_type_switch=int(np.sum(whole_sigmoid_crossing[index] & kind_switch[index])),
        ))
    report = dict(
        purpose='read-only margin/risk-gradient coverage audit',
        residual=str(args.residual.resolve()),
        residual_sha256=hashlib.sha256(args.residual.read_bytes()).hexdigest(),
        inputs=str(args.inputs.resolve()),
        input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        risk=dict(kappa=risk.kappa, D0=risk.D0,
                  sigmoid_10_to_90_margin_interval=[float(low), float(high)],
                  maximum_abs_dr_dm=float(risk.kappa / (4 * risk.D0))),
        aggregate=dict(
            episodes=count,
            executed_steps=int(valid.sum()),
            finite_margin_count=int(finite.sum()),
            margin_quantiles=np.quantile(values, [.01, .1, .5, .9, .99]).tolist() if len(values) else [],
            consecutive_finite_margin_delta_quantiles=(
                np.quantile(finite_margin_delta, [.5, .9, .99]).tolist() if len(finite_margin_delta) else []),
            transition_fraction=fraction(transition[finite]),
            low_risk_saturation_fraction=fraction(low_saturation[finite]),
            high_risk_saturation_fraction=fraction(high_saturation[finite]),
            mean_abs_dr_dm=float(derivative.mean()) if len(derivative) else None,
            cone_type_switches=int(kind_switch.sum()),
            candidate_constraint_switches=int(candidate_switch.sum()),
            large_risk_changes=int(large.sum()),
            large_change_with_cone_switch=int(np.sum(large & any_kind_switch)),
            large_change_with_candidate_switch=int(np.sum(large & any_candidate_switch)),
            large_change_with_cone_risk_change=int(np.sum(large & cone_change)),
            large_change_with_activity_change=int(np.sum(large & activity_change)),
            large_change_adjacent_to_transition_margin=int(np.sum(large & was_transition)),
            whole_sigmoid_margin_crossings=int(whole_sigmoid_crossing.sum()),
            whole_crossing_with_cone_type_switch=int(np.sum(whole_sigmoid_crossing & kind_switch)),
        ),
        episodes=episodes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['aggregate'], indent=2))


if __name__ == '__main__':
    main()
