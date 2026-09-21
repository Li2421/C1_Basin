"""Full-episode directional derivative decomposition; never updates a model.

Detach individual risk inputs, preserving forward values and all upstream
state BPTT on each retained path. The three derivatives sum by the chain rule.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/c1-gradient-mpl')
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1 import episode_rollout as episode
from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.risk.evaluation import risk_from_metadata
from single_integrator.c1.socp import ExactProjection
from single_integrator.c1.train import rollout_terms
from single_integrator.environment import Config
from single_integrator.cbf import CBFConfig
from single_integrator.evaluate import load_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--residual', type=Path, default=ROOT/'results/c1_guarded_seed0/residual.pkl')
    p.add_argument('--inputs', type=Path, default=ROOT/'results/c1_dataset_frozen_sets/calibration.npz')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=2)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    jax.config.update('jax_enable_x64', True)
    saved = pickle.loads(args.residual.read_bytes())
    meta = saved['metadata']
    risk = risk_from_metadata(meta)
    baseline, provenance = load_policy(Path(meta['baseline_checkpoint']))
    plant = Config(**provenance['evaluation_environment'])
    model = ResidualCorrection(hidden_dims=tuple(meta['architecture']['hidden_dims']),
                               layer_norm=meta['architecture']['layer_norm'])
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), saved['params'])
    initial = model.init(jax.random.PRNGKey(meta['training']['seed']),
                         jnp.zeros((1,4)), jnp.zeros((1,1)), jnp.zeros((1,20)))
    # A unitless scale along the actual accumulated optimizer displacement.
    direction = jax.tree_util.tree_map(lambda a,b: a-b, params, initial)
    field = ResidualFlowField(baseline, model)
    field.baseline_sample = jax.jit(field.baseline_sample)
    field.correction = jax.jit(field.correction)
    projection = ExactProjection()
    with np.load(args.inputs, allow_pickle=False) as data:
        starts = jnp.asarray(data['initial_positions'][:args.episodes], jnp.float64)
        noise = jnp.asarray(data['noise'][:args.episodes])
    if len(starts) != args.episodes or args.episodes < 1:
        raise ValueError('invalid episode count')
    original_soft, original_score = episode.soft_risk_diagnostics, episode.trajectory_risk_v2
    report = dict(residual=str(args.residual), sha256=hashlib.sha256(args.residual.read_bytes()).hexdigest(),
                  inputs=str(args.inputs), inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                  input_indices=list(range(args.episodes)), direction='phi + scale*(phi - initialization)',
                  direction_norm=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(direction)))),
                  scope='full BPTT directional derivatives, fixed selected event/topology branches', paths={}, finite_differences=[])

    def save():
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def objective(scale, details=False):
        shifted = jax.tree_util.tree_map(lambda a,d:a+scale*d, params, direction)
        result = rollout_terms(shifted, field, projection, starts, noise, plant, CBFConfig(), risk,
                               return_details=details)
        return result if details else jnp.mean(result[2])

    for path in ('activity', 'cone', 'task', 'total'):
        def soft(*a, **kw):
            d = original_soft(*a, **kw)
            activity, cone = d['local_activity'], d['cone_risk']
            if path not in ('activity', 'total'):
                activity = jax.lax.stop_gradient(activity)
            if path not in ('cone', 'total'):
                cone = jax.lax.stop_gradient(cone)
            return dict(d, risk=jnp.max(activity*cone))

        def score(c, positions, *a, **kw):
            if path not in ('task', 'total'):
                positions = jax.lax.stop_gradient(positions)
            if path == 'task':
                c = jax.lax.stop_gradient(c)
            return original_score(c, positions, *a, **kw)

        episode.soft_risk_diagnostics, episode.trajectory_risk_v2 = soft, score
        value, grad = jax.value_and_grad(objective)(jnp.array(0., jnp.float64))
        report['paths'][path] = dict(value=float(value), directional_derivative=float(grad))
        save()
        print(path, report['paths'][path], flush=True)
        jax.clear_caches()
    episode.soft_risk_diagnostics, episode.trajectory_risk_v2 = original_soft, original_score
    parts = sum(report['paths'][k]['directional_derivative'] for k in ('activity','cone','task'))
    report['chain_rule_sum_error'] = abs(parts-report['paths']['total']['directional_derivative'])
    np.testing.assert_allclose(parts, report['paths']['total']['directional_derivative'], rtol=1e-5, atol=1e-7)
    base = objective(jnp.array(0.), details=True)
    d = jax.device_get(base[3])
    valid = d['episode_mask']
    np.savez_compressed(args.out/'trace.npz', **{k: d[k] for k in (
        'margins','cone_types','local_activity','cone_risk','risk','episode_mask',
        'residual','h','terminal_codes','episode_lengths','stall_risk_t')})
    for step in (1e-2, 1e-3, 1e-4):
        endpoints=[]
        for sign in (-1,1):
            result = objective(jnp.array(sign*step), details=True)
            extra = jax.device_get(result[3])
            endpoints.append(dict(value=float(jnp.mean(result[2])),
                lengths=extra['episode_lengths'].tolist(), codes=extra['terminal_codes'].tolist(),
                changed_cone_entries=int(np.sum((extra['cone_types']!=d['cone_types']) & valid[...,None] & extra['episode_mask'][...,None]))))
        fd = (endpoints[1]['value']-endpoints[0]['value'])/(2*step)
        report['finite_differences'].append(dict(step=step, derivative=fd, endpoints=endpoints,
            absolute_error=abs(fd-report['paths']['total']['directional_derivative'])))
        save()
        print('FD', report['finite_differences'][-1], flush=True)
    report['base'] = dict(lengths=d['episode_lengths'].tolist(), codes=d['terminal_codes'].tolist())
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
