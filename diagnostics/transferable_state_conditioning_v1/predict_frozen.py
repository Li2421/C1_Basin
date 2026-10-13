"""Freeze source-selected baselines and predict final panels without labels.

This module reads proposal/state manifests and source-only checkpoints. It
never opens held-out rollout outcomes or the shared rollout database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import EtaOnly as WideEtaOnly
from diagnostics.physical_state_critic_formal_v1 import physical_entities
from diagnostics.physical_state_critic_formal_v1.train_unified import select_key
from diagnostics.transferable_state_conditioning_v1.assemble_source import (
    HERE, FORMAL, PANELS, ETA_CENTER, ETA_RADIUS, read, sha,
)
from diagnostics.transferable_state_conditioning_v1.train_baselines import SEEDS, NEAR_TIE

_LOGIT_CACHE = {}


def freeze():
    result = {"schema": "source_only_baseline_models_frozen_v1",
              "checkpoint_rule": "source-validation selection regret, pairwise and NLL near-ties",
              "final_outcomes_opened": False, "panels": {}}
    for panel in PANELS:
        picked = {}
        for kind in ("state_concat", "eta_only_wide"):
            runs = [read(HERE / "fits" / panel / kind / f"seed{s}" / "complete.json") for s in SEEDS]
            best = min(runs, key=lambda r: (*select_key(r["selected"], NEAR_TIE), r["seed"]))
            checkpoint = HERE / "fits" / panel / kind / f"seed{best['seed']}" / "checkpoint.msgpack"
            assert sha(checkpoint) == best["checkpoint_sha256"]
            picked[kind] = {"seed": best["seed"], "checkpoint": str(checkpoint),
                            "checkpoint_sha256": best["checkpoint_sha256"],
                            "source_data_sha256": best["trainval_sha256"],
                            "source_validation": best["selected"]}
        result["panels"][panel] = picked
    target = HERE / "baseline_models_frozen.json"
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if target.exists():
        assert target.read_text() == payload, "Frozen models may not change after confirmation"
    else:
        target.write_text(payload)
    print(target, sha(target))


def target_data(scene, bank):
    split = read(HERE / "frozen_splits.json")
    all_rows = read(FORMAL / "loso" / scene / "test" / "proposals.json")["states"]
    ids = split["final_confirmation"][scene]["state_uids"] if bank == "proposal" else split["anchor_subset"][scene]["state_uids"]
    proposal = {r["state_uid"]: r for r in all_rows}
    state_file = read(FORMAL / f"states_{scene}.json")["states"]
    by_alias = {r["alias"]: r for r in state_file}
    physical = [by_alias[proposal[uid]["state_alias"]]["physical"] for uid in ids]
    x = physical_entities.batch([{"scene": scene, "physical": p} for p in physical])
    if bank == "proposal":
        eta = np.asarray([proposal[uid]["eta"] for uid in ids], np.float32)
    else:
        fixed = np.asarray(split["common_anchor_eta"], np.float32)
        eta = np.broadcast_to(fixed, (len(ids), 16, 3)).copy()
    assert eta.shape == (len(ids), 16, 3)
    return ids, x, eta


def logits(info, kind, x_numpy, eta_numpy, mapped=None):
    key = (info["checkpoint_sha256"], kind, id(x_numpy), id(eta_numpy))
    if key not in _LOGIT_CACHE:
        x = {k: jnp.asarray(v) for k, v in x_numpy.items()}
        eta = jnp.asarray((eta_numpy-ETA_CENTER)/ETA_RADIUS)
        net = rep.Critic() if kind == "state_concat" else WideEtaOnly()
        template = net.init(jax.random.PRNGKey(info["seed"]),
                            {k:v[:1] for k,v in x.items()}, eta[:1,:1].reshape(1,3))
        ck = Path(info["checkpoint"])
        assert sha(ck) == info["checkpoint_sha256"]
        params = serialization.from_bytes(template, ck.read_bytes())
        @jax.jit
        def apply_batch(xx, ee):
            return net.apply(params, xx, ee)
        _LOGIT_CACHE[key] = (x_numpy, eta_numpy, x, eta, apply_batch)
    _, _, x, eta, apply_batch = _LOGIT_CACHE[key]
    n = len(eta_numpy)
    mapping = np.arange(n) if mapped is None else np.asarray(mapped, int)
    assert mapping.shape == (n,)
    xi = np.repeat(mapping, 16)
    ei = np.repeat(np.arange(n), 16)
    jj = np.tile(np.arange(16), n)
    chunks = []
    for a in range(0, len(xi), 512):
        ix = jnp.asarray(xi[a:a+512]); ie = jnp.asarray(ei[a:a+512]); jx = jnp.asarray(jj[a:a+512])
        chunks.append(np.asarray(apply_batch({k:v[ix] for k,v in x.items()}, eta[ie,jx])))
    out = np.concatenate(chunks).reshape(n,16)
    assert np.isfinite(out).all()
    return out


def derangement(n, rng):
    order = rng.permutation(n)
    p = np.empty(n, int)
    p[order] = np.roll(order, int(rng.integers(1,n)))
    assert np.all(p != np.arange(n))
    return p


def predict(panel, bank):
    assert panel in PANELS and bank in ("proposal", "anchor")
    freeze_file = HERE / "baseline_models_frozen.json"
    models = read(freeze_file)["panels"][panel]
    scene = panel.removeprefix("within_").removeprefix("loso_")
    ids, x, eta = target_data(scene, bank)
    z_state = logits(models["state_concat"], "state_concat", x, eta)
    z_eta = logits(models["eta_only_wide"], "eta_only_wide", x, eta)
    rng = np.random.default_rng(20261008)
    permutations = np.stack([derangement(len(ids), rng) for _ in range(100)])
    mismatch = np.stack([logits(models["state_concat"], "state_concat", x, eta, p).argmax(1)
                         for p in permutations]).astype(np.int16)
    output = HERE / "predictions" / panel / f"{bank}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, state_uids=np.asarray(ids), eta=eta,
                        state_logits=z_state, eta_logits=z_eta,
                        state_choice=z_state.argmax(1), eta_choice=z_eta.argmax(1),
                        mismatch_choice=mismatch, state_permutations=permutations)
    audit = {"panel": panel, "bank": bank, "states": len(ids),
             "models_frozen_sha256": sha(freeze_file),
             "predictions_sha256": sha(output),
             "final_outcomes_opened": False,
             "state_selection_entropy": float(-np.sum((lambda p:p[p>0]*np.log2(p[p>0]))(
                 np.bincount(z_state.argmax(1), minlength=16)/len(ids))))}
    (output.parent / f"{bank}_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("freeze", "predict"))
    p.add_argument("--panel", choices=tuple(PANELS))
    p.add_argument("--bank", choices=("proposal", "anchor"))
    a = p.parse_args()
    if a.action == "freeze":
        freeze()
    else:
        assert a.panel and a.bank
        predict(a.panel, a.bank)
