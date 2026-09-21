"""Load a scene-matched Stage-I baseline after checking its frozen manifest."""
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp

from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.train_deadlock_primary import digest
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config
from single_integrator.evaluate import ROOT, load_policy


def setup(manifest, seed=0):
    manifest = Path(manifest)
    frozen = json.loads(manifest.read_text())
    checkpoint = Path(frozen['checkpoint'])
    protocol_path = manifest.parent.parent/'protocol.json'
    if digest(checkpoint) != frozen['sha256'] or digest(protocol_path) != frozen['stage_i_protocol_sha256']:
        raise ValueError('Frozen Stage-I checkpoint/protocol changed')
    protocol = json.loads(protocol_path.read_text())
    if protocol['version'] not in ('matched_scene_stage_i_v1', 'matched_scene_stage_i_wide_v1'):
        raise ValueError('Unknown baseline protocol')
    if any(digest(ROOT/path) != sha for path,sha in protocol['source_sha256'].items()):
        raise ValueError('Stage-I source changed')
    saved = pickle.loads(checkpoint.read_bytes())
    metrics = [json.loads(line) for line in (checkpoint.parent/'metrics.jsonl').read_text().splitlines()]
    selected = min(metrics, key=lambda row: row['selection_val_loss'])
    if saved['step'] != selected['step'] or max(row['step'] for row in metrics) != protocol['steps']:
        raise ValueError('Baseline was not selected by the full original Stage-I budget')
    jax.config.update('jax_enable_x64', True)
    baseline, provenance = load_policy(checkpoint)
    if provenance['evaluation_environment'] != frozen['environment']:
        raise ValueError('Baseline environment mismatch')
    plant = Config(**frozen['environment'])
    model = ResidualCorrection(hidden_dims=tuple(baseline.config['actor_hidden_dims']),
        layer_norm=baseline.config['actor_layer_norm'])
    params = model.init(jax.random.PRNGKey(seed),jnp.zeros((1,4)),jnp.zeros((1,1)),jnp.zeros((1,20)))
    params = jax.tree_util.tree_map(lambda x:jnp.asarray(x,jnp.float64),params)
    field = ResidualFlowField(baseline,model)
    field.baseline_sample = jax.jit(field.baseline_sample)
    field.correction = jax.jit(field.correction)
    return params,field,plant,CBFConfig(),checkpoint
