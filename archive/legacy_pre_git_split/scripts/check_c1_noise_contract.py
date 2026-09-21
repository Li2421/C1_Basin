"""Check original Flow-BC sampling against the frozen differentiable interface."""
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.train_completion import setup, noise
from single_integrator.environment import GiveWayEnv


def main():
    records = []
    for baseline_seed in (0, 1):
        _, field, plant, _, _ = setup(baseline_seed=baseline_seed)
        env = GiveWayEnv(plant)
        for rid, initial in enumerate([[[-.6, .003], [.9, -.004]], [[-.95, -.01], [.58, .02]]]):
            env.reset(initial)
            obs = jnp.asarray(env.observation()[None])
            draws = noise(42, rid)
            for step in [0, 19, 420, 849]:
                key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42), rid), step)
                with jax.experimental.disable_x64():
                    original_noise = jax.random.normal(key, (1, 4))
                    original = field.baseline.sample_actions(obs, seed=key).reshape(1, 4)
                reproduced = field.baseline_sample(obs, draws[step:step+1])
                assert np.array_equal(original_noise, draws[step:step+1])
                difference = float(np.max(np.abs(np.asarray(original)-np.asarray(reproduced))))
                assert difference <= 1e-6, difference
                records.append(dict(baseline=baseline_seed, rid=rid, step=step, max_difference=difference))
    path = ROOT/'results/c1_independent_distribution/noise_contract.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(dict(scope='Nominal sample parity only; does not substitute for full Safety benchmark reproduction',
        records=records), indent=2)+'\n')
    print(dict(samples=len(records), max_difference=max(r['max_difference'] for r in records)))


if __name__ == '__main__':
    main()
