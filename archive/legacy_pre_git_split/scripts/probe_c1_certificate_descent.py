"""Full hard-projection BPTT and actual guarded descent on a known deadlock."""
import argparse
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.train_v3 import setup, noise
from single_integrator.c1.rollout_certificate import rollout
from single_integrator.c1.training.step_control import guarded_step, finite
from single_integrator.c1.training.persistence import atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    case = ROOT/'results/c1_four_objectives_multiseed/evaluation/baseline/400005_20260916.npz'
    with np.load(case) as z:
        initial = jnp.asarray(z['positions_before'][0])
    params, field, plant, cbf, _ = setup()
    draws = jnp.concatenate([noise(20260915, 400005)[:100], noise(20260916, 400005)[100:]])
    fn = jax.jit(lambda phi: rollout(phi, field, initial, draws, plant, cbf)[0])
    def objective(phi):
        r = fn(phi)
        return r['J_live']+r['J_def'], r
    started = time.monotonic()
    before, grad = jax.value_and_grad(objective, has_aux=True)(params)
    assert finite((before, grad))
    optimizer = optax.adam(1e-5)
    new, _, after, decision = guarded_step(params, optimizer.init(params), grad, optimizer, objective, before)
    result = dict(scope='Already-seen local fixed-lambda diagnostic; not constrained training or efficacy',
        case=str(case), before={k: float(v) for k, v in before[1].items()},
        after={k: float(v) for k, v in after[1].items()}, gradient_norm=float(optax.global_norm(grad)),
        step_control=decision, elapsed_seconds=time.monotonic()-started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(args.out, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
