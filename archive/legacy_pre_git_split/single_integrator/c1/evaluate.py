"""Evaluate a trained C1 residual under the selected 309/314 protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.risk.evaluation import risk_from_metadata, audit_trace, aggregate_risk
from single_integrator.c1.train import approved_checkpoint
from single_integrator.cbf import CBFConfig, CBFSafetyFilter, barrier_constraints, project_velocity
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.evaluate import load_policy, rollout
from single_integrator.outcomes import first_event_aggregate


class C1Policy:
    """Runtime residual policy; the usual CBF filter remains downstream."""
    def __init__(self, field, params, plant):
        self.field, self.params = field, jax.tree_util.tree_map(jnp.asarray, params)
        self.plant, self.env, self.cbf = plant, GiveWayEnv(plant), CBFConfig()
        self.records = []
        self.field.baseline_sample = jax.jit(self.field.baseline_sample)
        self.field.correction = jax.jit(self.field.correction)

    def sample_actions(self, observations, seed):
        noise = jax.random.normal(seed, (len(observations), 4), dtype=jnp.float32)
        positions = np.asarray(observations)[0, :, :2]
        A, b, _ = barrier_constraints(dict(positions=positions, walls=self.env.walls,
                                          config=self.plant.to_dict()), self.cbf)
        def projection(target, A, b, speed):
            return jnp.asarray(project_velocity(np.asarray(target)[0], A, b, speed, self.cbf)[0].reshape(1, 4))
        result = self.field.prepare(self.params, observations, noise, A, b, self.plant.max_speed, projection)
        self.records.append({'c1_'+k: np.asarray(v)[0] for k, v in result.items()})
        return result['candidate'].reshape(-1, 2, 2)

    def nominal_control(self, raw, max_speed):
        # C1 candidate must reach the final projection without radial clipping.
        return np.asarray(raw, dtype=np.float64)


def main():
    jax.config.update('jax_enable_x64', True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--residual', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--initial-states', type=Path, required=True)
    p.add_argument('--split', choices=('test', 'val'), default='test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-rollouts', type=int,
                   help='Evaluate a prefix of the fixed 200-start suite; default is all 200.')
    p.add_argument('--rollout-ids', type=int, nargs='+',
                   help='Diagnostic subset of fixed-suite IDs; preserves original Flow noise IDs.')
    p.add_argument('--max-steps', type=int,
                   help='Diagnostic-only extension of the frozen evaluation horizon.')
    args = p.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('out-dir must be empty')
    with args.residual.open('rb') as handle:
        saved = pickle.load(handle)
    metadata = saved['metadata']
    risk = risk_from_metadata(metadata)
    if metadata.get('method') != 'c1_v0':
        raise ValueError('Legacy Flow-field residual is not a C1 V0 checkpoint')
    checkpoint = Path(metadata['baseline_checkpoint'])
    if approved_checkpoint(checkpoint) != metadata['baseline_checkpoint_sha256']:
        raise ValueError('residual and selected baseline checkpoint differ')
    baseline, provenance = load_policy(checkpoint)
    plant = Config(**provenance['evaluation_environment'])
    if plant.to_dict() != metadata['environment']:
        raise ValueError('residual environment differs from its frozen baseline')
    if args.max_steps is not None:
        if args.max_steps < plant.max_steps:
            raise ValueError('diagnostic max-steps may only extend the frozen protocol')
        from dataclasses import replace
        plant = replace(plant, max_steps=args.max_steps)
    with np.load(args.initial_states) as data:
        initial = np.asarray(data[args.split + '_initial_positions'])
        suite_hash = hashlib.sha256(args.initial_states.read_bytes()).hexdigest()
    if suite_hash != '30a575df16d56b65cb92b97a95fbcad8b454f49621de45a4359e6bbf947090cb' or len(initial) != 200:
        raise ValueError('C1 evaluation requires the selected fixed 200-start suite')
    n_rollouts = len(initial) if args.n_rollouts is None else args.n_rollouts
    if not 1 <= n_rollouts <= len(initial):
        raise ValueError('n-rollouts must be in [1, 200] for the fixed suite')
    if args.rollout_ids is not None:
        if args.n_rollouts is not None:
            raise ValueError('choose either --n-rollouts or --rollout-ids')
        if len(set(args.rollout_ids)) != len(args.rollout_ids) or any(rid < 0 or rid >= len(initial) for rid in args.rollout_ids):
            raise ValueError('rollout IDs must be unique members of [0, 199]')
        rollout_ids = args.rollout_ids
        n_rollouts = len(rollout_ids)
    else:
        rollout_ids = list(range(n_rollouts))
    model = ResidualCorrection(hidden_dims=tuple(metadata['architecture']['hidden_dims']),
                               layer_norm=metadata['architecture']['layer_norm'])
    field = ResidualFlowField(baseline, model)
    policy, cbf = C1Policy(field, saved['params'], plant), CBFConfig()
    if metadata['cbf'] != cbf.to_dict():
        raise ValueError('residual CBF differs from the frozen baseline CBF')
    args.out_dir.mkdir(parents=True)
    summaries = []
    for rid in rollout_ids:
        start = initial[rid]
        policy.records.clear()
        summary, trace = rollout(policy, start, plant, args.seed, rid, CBFSafetyFilter(cbf), cbf_config=cbf)
        trace.update({key: np.stack([row[key] for row in policy.records]) for key in policy.records[0]})
        summary['c1'], diagnostics = audit_trace(trace, plant, cbf, risk)
        trace.update(diagnostics)
        summaries.append(summary)
        np.savez_compressed(args.out_dir/f'rollout_{rid:04d}.npz', **trace, initial_positions=start)
        print(f'c1 rollout={rid} outcome={summary["outcome"]} steps={summary["episode_steps"]}', flush=True)
    aggregate = dict(first_event_aggregate(summaries), c1=aggregate_risk(summaries))
    (args.out_dir/'summary.json').write_text(json.dumps(dict(aggregate=aggregate, rollouts=summaries), indent=2)+'\n')
    (args.out_dir/'config.json').write_text(json.dumps(dict(method='c1_v0', risk_version=metadata.get('risk_version', 'R_risk_v0'), hard_diagnostic_version='R_risk_v0', residual=str(args.residual.resolve()), residual_sha256=hashlib.sha256(args.residual.read_bytes()).hexdigest(), baseline_checkpoint=str(checkpoint), baseline_checkpoint_sha256=metadata['baseline_checkpoint_sha256'], environment=plant.to_dict(), cbf=cbf.to_dict(), initial_states=str(args.initial_states.resolve()), initial_states_sha256=suite_hash, seed=args.seed, split=args.split, n_rollouts=n_rollouts, diagnostic_horizon_extension=args.max_steps), indent=2)+'\n')
    (args.out_dir/'evaluated_ids.json').write_text(json.dumps(dict(
        rollout_ids=rollout_ids, diagnostic_subset=args.rollout_ids is not None), indent=2)+'\n')
    (args.out_dir/'complete.json').write_text(json.dumps(dict(completed_rollouts=len(summaries)))+'\n')


if __name__ == '__main__':
    main()
