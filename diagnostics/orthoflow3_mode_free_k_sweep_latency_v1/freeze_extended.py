#!/usr/bin/env python3
"""Extend the frozen generator's deterministic draw stream to K=16."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

ROOT = Path('/home/zhihan/research/Basin_C1')
OLD = ROOT / 'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location('prior_freeze', OLD / 'freeze_proposals.py')
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old = json.loads((OLD / 'frozen_proposals.json').read_text())
    training = json.loads((OLD / 'generator_training.json').read_text())
    h = np.load(OLD / 'cohort_features.npz')['h_raw']
    model = prior.genlib.Generator()
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 214), jnp.float32), 'Toy')
    params = serialization.from_bytes(template, Path(training['selected']['checkpoint']).read_bytes())
    norm = prior.load(prior.OLD / 'dataset_manifest.json')['state_normalization']['Toy']
    x = (h - np.asarray(norm['mean'], np.float32)) / np.asarray(norm['std'], np.float32)
    raw = np.asarray(model.apply(params, jnp.asarray(x), 'Toy'))
    mu, sigma = map(np.asarray, prior.genlib.dist_params(jnp.asarray(raw)))
    center, radius = prior.genlib.CENTER, prior.genlib.RADIUS
    means = center + radius * np.tanh(mu)
    samples = []
    for i in range(len(h)):
        rng = np.random.default_rng(int(hashlib.sha256(
            f"mode_free_v1|seed{training['selected']['seed']}|wide_ic|{i}".encode()).hexdigest()[:16], 16))
        z = rng.standard_normal((16, 3))
        samples.append(center + radius * np.tanh(mu[i] + sigma[i] * z))
    samples = np.asarray(samples)
    for i, state in enumerate(old['states']):
        if not np.allclose(means[i], state['eta']['generator_mean'], atol=1e-7, rtol=1e-7):
            raise RuntimeError(f'Frozen generator mean mismatch at state {i}')
        for j in range(4):
            if not np.allclose(samples[i,j], state['eta'][f'sample_{j}'], atol=1e-7, rtol=1e-7):
                raise RuntimeError(f'Frozen sample {j} mismatch at state {i}')
    scores, critic_used = prior.critic_scores(h, samples)
    for i, state in enumerate(old['states']):
        if not np.allclose(scores[i,:4], state['critic_scores'], atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'Frozen critic score mismatch at state {i}')
    states = []
    for i, state in enumerate(old['states']):
        row = {k: state[k] for k in ('episode_index','rollout_id','initial_positions','source_group','h_sha256')}
        row['eta'] = {f'sample_{j}': [float(v) for v in samples[i,j]] for j in range(4,16)}
        row['all_critic_scores'] = scores[i].tolist()
        row['sigma'] = sigma[i].tolist()
        states.append(row)
    payload = {'cohort': old['cohort'], 'cohort_sha256': old['cohort_sha256'],
               'source_frozen_sha256': sha(OLD / 'frozen_proposals.json'),
               'generator_training_sha256': sha(OLD / 'generator_training.json'),
               'critic_checkpoints': critic_used, 'K_max': 16,
               'proposal_seed_rule': old['proposal_seed_rule'],
               'selection_before_extended_outcomes': True, 'states': states}
    (OUT / 'frozen_proposals.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    np.savez_compressed(OUT / 'cohort_features.npz', h_raw=h)
    (OUT / 'freeze_manifest.json').write_text(json.dumps({
        'frozen_proposals_sha256': sha(OUT / 'frozen_proposals.json'),
        'source_frozen_sha256': payload['source_frozen_sha256'],
        'selected_generator_checkpoint_sha256': training['selected']['sha256'],
        'K_values': [1,2,4,8,16], 'old_first_four_verified': True,
        'critic_first_four_verified': True, 'test_outcomes_used': False,
    }, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'states': len(states), 'new_samples_per_state': 12,
                      'old_first_four_verified': True, 'critic_first_four_verified': True}))


if __name__ == '__main__': main()
