"""Probe stale optimizer momentum on a current already-seen successful start."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import optax
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.train_completion import setup, noise
from single_integrator.c1.rollout_completion import rollout
from single_integrator.c1.training.step_control import guarded_step as original
from single_integrator.c1.training.step_control_restart import guarded_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    blob = args.checkpoint.read_bytes()
    saved = pickle.loads(blob); config = saved['config']
    _, field, plant, cbf, _ = setup(config['seed'], config['baseline_seed'])
    params = jax.tree_util.tree_map(jnp.asarray, saved['params'])
    state = jax.tree_util.tree_map(jnp.asarray, saved['optimizer'])
    data = json.loads(Path(config['sets']).read_text())
    fn = jax.jit(lambda phi, x, n: rollout(phi, field, x, n, plant, cbf, prefix_steps=0)[0])
    chosen = None
    for item in data['pools']['train']:
        draws = noise(data['calibration_noise_seeds'][0], item['rid'])
        initial = jnp.asarray(item['initial'])
        terms = fn(params, initial, draws)
        if float(terms['J_live']) == 0 and bool(terms['success']):
            chosen = item
            break
    if chosen is None:
        raise ValueError('no zero-risk development trajectory; cannot diagnose the feasible phase')
    def objective(phi):
        r = fn(phi, initial, draws)
        return r['J_def']+saved['dual']*(r['J_live']-saved['epsilon']), r
    before, gradient = jax.value_and_grad(objective, has_aux=True)(params)
    optimizer = optax.adam(config['lr'])
    legacy = original(params, state, gradient, optimizer, objective, before)
    restart = guarded_step(params, state, gradient, optimizer, objective, before)
    result = dict(checkpoint_sha256=hashlib.sha256(blob).hexdigest(), update=saved['completed_updates'],
        rid=chosen['rid'], scope='Selected successful TRAIN trajectory; optimizer diagnostic, no policy deployment',
        before={k: float(v) for k, v in before[1].items()},
        legacy=legacy[3], restart=restart[3],
        after={k: float(v) for k, v in restart[2][1].items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
