"""Source-validation selection and same-scene state mismatch diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import EtaOnly as WideEtaOnly
from diagnostics.physical_state_critic_formal_v1.train_unified import select_key
from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, PANELS, read, sha
from diagnostics.transferable_state_conditioning_v1.train_baselines import SEEDS, NEAR_TIE

HERE = Path(os.environ.get("TRANSFER_Q_WORK_ROOT", str(HERE)))
_PRED_CACHE = {}


def bootstrap(values, seed=20261008):
    a = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(len(a), size=(10000, len(a)))].mean(1)
    return {"n": len(a), "mean": float(a.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def best_run(panel, kind):
    fits = [read(HERE / "fits" / panel / kind / f"seed{seed}" / "complete.json") for seed in SEEDS]
    selected = min(fits, key=lambda r: (*select_key(r["selected"], NEAR_TIE), r["seed"]))
    return selected


def predict(panel, kind, run, state_index=None):
    key = (panel, kind, run["seed"])
    if key not in _PRED_CACHE:
        d = np.load(HERE / "source_data" / panel / "trainval.npz")
        val = np.flatnonzero(d["split"] == "validation")
        x = {name[7:]: jnp.asarray(d[name][val]) for name in d.files if name.startswith("entity_")}
        eta = jnp.asarray(d["eta"][val])
        net = rep.Critic() if kind == "state_concat" else WideEtaOnly()
        template = net.init(jax.random.PRNGKey(run["seed"]),
                            {name: value[:1] for name, value in x.items()}, eta[:1, :1].reshape(1, 3))
        ck = HERE / "fits" / panel / kind / f"seed{run['seed']}" / "checkpoint.msgpack"
        assert sha(ck) == run["checkpoint_sha256"]
        params = serialization.from_bytes(template, ck.read_bytes())
        @jax.jit
        def apply_batch(xx, ee):
            return net.apply(params, xx, ee)
        _PRED_CACHE[key] = (x, eta, apply_batch)
    x, eta, apply_batch = _PRED_CACHE[key]
    n = eta.shape[0]
    mapped = np.arange(n) if state_index is None else np.asarray(state_index, int)
    assert mapped.shape == (n,)
    ii = np.repeat(mapped, 16)
    jj = np.tile(np.arange(16), n)
    ei = np.repeat(np.arange(n), 16)
    scores = []
    for start in range(0, len(ii), 512):
        ix = jnp.asarray(ii[start:start+512]); ex = jnp.asarray(ei[start:start+512]); jx = jnp.asarray(jj[start:start+512])
        z = apply_batch({k: v[ix] for k, v in x.items()}, eta[ex, jx])
        scores.append(np.asarray(z))
    return np.concatenate(scores).reshape(n, 16)


def evaluate(panel):
    d = np.load(HERE / "source_data" / panel / "trainval.npz")
    val = np.flatnonzero(d["split"] == "validation")
    scenes = np.asarray(d["scene"])[val]
    exposure = d["success"][val] + d["failure"][val] + d["numerical"][val]
    assert np.all(exposure == exposure.flat[0])
    nseeds = int(exposure.flat[0])
    assert nseeds in (4, 16)
    true_q = np.asarray(d["success"])[val] / nseeds
    true_b15 = np.asarray(d["success"])[val] >= (15 if nseeds == 16 else 4)
    state_run = best_run(panel, "state_concat")
    eta_run = best_run(panel, "eta_only_wide")
    z_state = predict(panel, "state_concat", state_run)
    z_eta = predict(panel, "eta_only_wide", eta_run)
    state_choice, eta_choice = z_state.argmax(1), z_eta.argmax(1)
    row = np.arange(len(val))
    rng = np.random.default_rng(20261008)
    mismatch_choice = np.empty((100, len(val)), np.int16)
    for t in range(100):
        mapped = np.arange(len(val))
        for scene in sorted(set(scenes)):
            ix = np.flatnonzero(scenes == scene)
            assert len(ix) > 1
            order = rng.permutation(ix)
            mapped[order] = np.roll(order, rng.integers(1, len(ix)))
            assert np.all(mapped[ix] != ix)
        mismatch_choice[t] = predict(panel, "state_concat", state_run, mapped).argmax(1)
    matched_q = true_q[row, state_choice]
    eta_q = true_q[row, eta_choice]
    mismatch_q = true_q[np.arange(len(val))[None, :], mismatch_choice]
    matched_b15 = true_b15[row, state_choice]
    eta_b15 = true_b15[row, eta_choice]
    mismatch_b15 = true_b15[np.arange(len(val))[None, :], mismatch_choice]
    result = {"schema": "source_only_validation_mismatch_v1", "panel": panel,
              "rollout_seeds_per_pair": nseeds,
              "robust_threshold_interpretation": "B15" if nseeds == 16 else "4/4 provisional, not B15",
              "validation_states": len(val), "source_scenes": list(PANELS[panel]),
              "state_run": state_run, "eta_run": eta_run,
              "matched_state_B15": int(matched_b15.sum()),
              "eta_only_B15": int(eta_b15.sum()),
              "mismatched_state_B15_mean": float(mismatch_b15.sum(1).mean()),
              "matched_state_mean_success": float(matched_q.mean()),
              "eta_only_mean_success": float(eta_q.mean()),
              "mismatched_state_mean_success": float(mismatch_q.mean()),
              "state_minus_eta_B15": bootstrap(matched_b15.astype(float)-eta_b15.astype(float)),
              "state_minus_mismatch_B15": bootstrap((matched_b15[None,:].astype(float)-mismatch_b15.astype(float)).mean(0)),
              "state_minus_mismatch_mean_success": bootstrap((matched_q[None,:]-mismatch_q).mean(0)),
              "matched_state_choice": state_choice.tolist(),
              "eta_choice": eta_choice.tolist(),
              "mismatch_choice": mismatch_choice.tolist(),
              "validation_state_uids": np.asarray(d["state_uids"])[val].tolist(),
              "source_data_sha256": sha(HERE / "source_data" / panel / "trainval.npz"),
              "test_labels_opened": False}
    output = HERE / "fits" / panel / "source_validation_diagnostic.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:result[k] for k in ("panel","validation_states","matched_state_B15","eta_only_B15","mismatched_state_B15_mean","state_minus_mismatch_B15")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", choices=tuple(PANELS), required=True)
    evaluate(p.parse_args().panel)
