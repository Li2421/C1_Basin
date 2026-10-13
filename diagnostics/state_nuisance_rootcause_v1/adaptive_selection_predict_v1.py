"""Frozen Q(s,eta), Q(eta), and 32-state diagnostic predictions on fixed states."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from ring_exchange.local_frame import vectors_to_local
from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import model

PRIOR = HERE / 'real_state_expansion'
OUT = HERE / 'heldout_adaptive_selection_v1'
SEEDS = (17, 23, 41)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw23(states):
    def wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    result = []
    for state in states:
        physical = state['physical']
        assert physical['timestep'] == 0
        positions, velocities, goals = [np.asarray(physical[key])
                                        for key in ('positions', 'velocities', 'goals')]
        theta = np.arctan2(positions[:, 1], positions[:, 0])
        goal_theta = np.arctan2(goals[:, 1], goals[:, 0])
        radii = np.linalg.norm(positions, axis=1)
        goal_radii = np.linalg.norm(goals, axis=1)
        relative = wrap(theta[1:] - theta[0] - np.arange(1, 4) * np.pi / 2)
        goal_delta = wrap(goal_theta - theta - np.pi)
        local_vel = vectors_to_local(velocities, positions)
        result.append(np.r_[radii, relative, goal_radii, goal_delta, local_vel.ravel()])
    return np.asarray(result, np.float32)


def main():
    import jax
    import jax.numpy as jnp
    from flax import serialization

    assert jax.default_backend() == 'gpu', 'Use the same GPU backend as the frozen original evaluation'

    protocol = json.loads((OUT / 'protocol.json').read_text())
    plan = json.loads((OUT / 'analysis_plan.json').read_text())
    prior_protocol = json.loads((PRIOR / 'protocol.json').read_text())
    frozen = json.loads((PRIOR / 'models_frozen.json').read_text())
    inputs = np.load(PRIOR / 'physical_inputs.npz')
    assert sha(PRIOR / 'models_frozen.json') == protocol['frozen_models_sha256']
    assert sha(PRIOR / 'physical_inputs.npz') == protocol['frozen_inputs_sha256']
    assert sha(OUT / 'analysis_plan.json') == protocol['analysis_plan_sha256']
    assert np.array_equal(np.asarray(protocol['eta']), np.asarray(prior_protocol['eta']))
    assert plan['physical_state_count'] == len(protocol['states']) == 256
    np.testing.assert_array_equal(raw23(prior_protocol['states']), inputs['raw_physical23'])

    raw = raw23(protocol['states'])
    state = (raw - inputs['physical_center']) / inputs['physical_scale']
    assert state.shape == (256, 23) and np.isfinite(state).all()
    eta = np.asarray(inputs['eta'])
    assert eta.shape == (16, 3)
    network = model('local_state_eta')
    template = network.init(jax.random.PRNGKey(17), jnp.asarray(state[:1]),
                            jnp.zeros((1, 3)), jnp.zeros((1, 76)))

    @jax.jit
    def predict(parameters, state_batch):
        i = jnp.repeat(jnp.arange(len(state_batch)), 16)
        j = jnp.tile(jnp.arange(16), len(state_batch))
        return network.apply(parameters, state_batch[i], jnp.asarray(eta)[j],
                             jnp.zeros((len(i), 76))).reshape(len(state_batch), 16)

    logits = {}
    model_fingerprints = {}
    prior_prediction_max_abs_error = 0.
    prior_fresh_indices = np.load(PRIOR / 'fresh_test_dataset.npz')['state_indices']
    previous_logits = np.load(PRIOR / 'evaluation_logits.npz')
    for name in ('physical23_eta_n512', 'physical23_eta_n32'):
        selection = frozen['selections'][name]
        predictions = []
        model_fingerprints[name] = {}
        for seed in SEEDS:
            checkpoint = next(m for m in selection['models'] if m['seed'] == seed)
            path = Path(checkpoint['path'])
            assert sha(path) == checkpoint['sha256']
            parameters = serialization.from_bytes(template, path.read_bytes())
            earlier = np.asarray(predict(parameters,
                                         jnp.asarray(inputs['physical23'][prior_fresh_indices])))
            expected = previous_logits[
                f"{name}__{seed}__{selection['selected_step']}__fresh_test"]
            np.testing.assert_allclose(earlier, expected, rtol=1e-5, atol=1e-5)
            prior_prediction_max_abs_error = max(
                prior_prediction_max_abs_error, float(np.max(np.abs(earlier - expected))))
            predictions.append(np.asarray(predict(parameters, jnp.asarray(state))))
            model_fingerprints[name][str(seed)] = checkpoint['sha256']
        logits[name] = np.stack(predictions)
        assert logits[name].shape == (3, 256, 16)

    old_train = np.load(PRIOR / 'source_dataset.npz')
    success = old_train['success'][:512]
    failure = old_train['failure'][:512]
    rate = success.sum(0) / (success + failure).sum(0)
    eta_only = np.log(np.clip(rate, 1e-6, 1-1e-6) /
                      (1 - np.clip(rate, 1e-6, 1-1e-6)))
    prior_logits = np.load(PRIOR / 'evaluation_logits.npz')['eta_only_n512__fresh_test']
    np.testing.assert_allclose(eta_only, prior_logits[0], rtol=0, atol=1e-12)
    assert np.all(prior_logits == prior_logits[0])
    assert int(np.argmax(eta_only)) == 13

    np.savez_compressed(OUT / 'frozen_predictions.npz',
                        state23=state, raw_state23=raw,
                        physical23_eta_n512_logits=logits['physical23_eta_n512'],
                        physical23_eta_n32_logits=logits['physical23_eta_n32'],
                        eta_only_n512_logits=eta_only,
                        eta=np.asarray(protocol['eta']), seeds=np.asarray(SEEDS),
                        state_uids=np.asarray([s['state_uid'] for s in protocol['states']]))
    audit = dict(states=256, etas=16, seeds=list(SEEDS),
                 inference_backend=jax.default_backend(),
                 prior_state_feature_parity_exact=True,
                 prior_eta_only_logit_parity_exact=True,
                 old_fresh_test_logit_parity_max_abs_error=prior_prediction_max_abs_error,
                 no_new_training=True,
                 prediction_file_sha256=sha(OUT / 'frozen_predictions.npz'),
                 protocol_sha256=sha(OUT / 'protocol.json'),
                 models_frozen_sha256=sha(PRIOR / 'models_frozen.json'),
                 input_normalizer_sha256=sha(PRIOR / 'physical_inputs.npz'),
                 checkpoint_hashes=model_fingerprints)
    (OUT / 'prediction_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(states=256, etas=16,
                          eta_only_choice=int(np.argmax(eta_only)),
                          prediction_file_sha256=audit['prediction_file_sha256'])))


if __name__ == '__main__':
    main()
