"""Freeze Ring FF TEST choices without reading any TEST outcomes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.state_nuisance_rootcause_v1.adaptive_selection_predict_v1 import raw23
from diagnostics.physical_state_critic_formal_v1.train_ring_ff import StateCritic, EtaCritic

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'state_nuisance_rootcause_v1' / 'real_state_expansion'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = json.loads((HERE / 'ring_ff_models_frozen.json').read_text())
    protocol = json.loads((HERE / 'matched_protocol.json').read_text())
    ring = [s for s in protocol['states'] if s['scenario'] == 'ring_exchange']
    assert len(ring) == 672
    assert len({s['state_uid'] for s in ring}) == 672
    inputs = np.load(PRIOR / 'physical_inputs.npz')
    trainval = np.load(HERE / 'ring_ff_trainval.npz')
    normalized = (raw23(ring) - inputs['physical_center']) / inputs['physical_scale']
    np.testing.assert_array_equal(normalized[:544], trainval['physical23'])
    np.testing.assert_array_equal(inputs['eta'], trainval['eta'])
    assert np.isfinite(normalized).all()
    x = jnp.asarray(normalized[544:])
    eta = jnp.asarray(trainval['eta'])
    ii = jnp.repeat(jnp.arange(128), 16)
    jj = jnp.tile(jnp.arange(16), 128)
    logits = {}
    checkpoint_hashes = {}
    for kind, network in [('state', StateCritic()), ('eta_only', EtaCritic())]:
        row = frozen['selected'][kind]
        path = HERE / 'ring_ff_fits' / kind / f"seed{row['seed']}" / 'checkpoint.msgpack'
        assert sha(path) == row['checkpoint_sha256']
        template = (network.init(jax.random.PRNGKey(row['seed']), x[:1], eta[:1])
                    if kind == 'state' else network.init(jax.random.PRNGKey(row['seed']), eta[:1]))
        parameters = serialization.from_bytes(template, path.read_bytes())
        values = (network.apply(parameters, x[ii], eta[jj]) if kind == 'state'
                  else network.apply(parameters, eta[jj]))
        logits[kind] = np.asarray(values).reshape(128, 16)
        assert np.isfinite(logits[kind]).all()
        checkpoint_hashes[kind] = sha(path)
    assert np.all(logits['eta_only'] == logits['eta_only'][0])
    target = HERE / 'ring_ff_test_predictions.npz'
    np.savez_compressed(target, state_logits=logits['state'],
                        eta_only_logits=logits['eta_only'],
                        state_choice=np.argmax(logits['state'], axis=1),
                        eta_only_choice=np.argmax(logits['eta_only'], axis=1),
                        state_uids=np.asarray([s['state_uid'] for s in ring[544:]]),
                        eta=np.asarray(protocol['eta']), normalized_eta=np.asarray(eta))
    audit = dict(schema='formal_q_ring_ff_predictions_v1', test_states=128,
                 checkpoint_hashes=checkpoint_hashes,
                 frozen_models_sha256=sha(HERE / 'ring_ff_models_frozen.json'),
                 matched_protocol_sha256=sha(HERE / 'matched_protocol.json'),
                 trainval_sha256=sha(HERE / 'ring_ff_trainval.npz'),
                 physical_normalizer_sha256=sha(PRIOR / 'physical_inputs.npz'),
                 test_labels_used=False, prediction_sha256=sha(target),
                 state_selection_counts=np.bincount(np.argmax(logits['state'], axis=1),
                                                    minlength=16).tolist(),
                 eta_only_choice=int(np.argmax(logits['eta_only'][0])),
                 backend=jax.default_backend())
    (HERE / 'ring_ff_test_prediction_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
