#!/usr/bin/env python3
"""Freeze proposals and critic decisions on the historical WIDE-IC cohort."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
OUT = Path(__file__).resolve().parent
GENSRC = ROOT / "diagnostics/orthoflow3_toy_db_conditional_generator_v1"
CRITIC = ROOT / "diagnostics/orthoflow3_nll_weighting_ablation_v1"
RANK = ROOT / "diagnostics/orthoflow3_ranking_aware_critic_v1"
OLD = ROOT / "diagnostics/orthoflow3_continuous_basin_critic_v1"
WIDE = ROOT / "diagnostics/gphi_wide_ic_cadence_v1"
TOY = ROOT / "diagnostics/orthoflow3_shared_eta_codebook_v1"
JOINT = ROOT / "diagnostics/orthoflow3_toy_db_joint_selector_v1"
DEFORM = ROOT / "diagnostics/orthoflow3_deformable_shared_modes_v1"
sys.path[:0] = [str(SYSROOT), str(ROOT)]

spec = importlib.util.spec_from_file_location("mode_free_gen", OUT / "train_generator.py")
genlib = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = genlib
spec.loader.exec_module(genlib)
spec = importlib.util.spec_from_file_location("ranklib", RANK / "run_experiment.py")
ranklib = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ranklib
spec.loader.exec_module(ranklib)
spec = importlib.util.spec_from_file_location("prior_genlib", GENSRC / "pipeline.py")
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())


def features_for_cohort(episodes, environment, cbf_config):
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
    from single_integrator.evaluate import load_policy
    config = Config(**environment)
    cbf = CBFConfig(**cbf_config)
    policy, provenance = load_policy(SYSROOT / "baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl")
    if provenance["evaluation_environment"] != environment:
        raise RuntimeError("Flow environment differs from historical benchmark")
    sample = jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0])
    builder = StartupAwareFeatureBuilder()
    outputs = []
    for ep in episodes:
        env = GiveWayEnv(config)
        env.reset(np.asarray(ep["initial_positions"], np.float64))
        episode_key = jax.random.fold_in(jax.random.PRNGKey(42), int(ep["rollout_id"]))
        key = jax.random.fold_in(episode_key, 0)
        obs = np.asarray(env.observation(), np.float32)
        flow = bounded_nominal(np.asarray(sample(jnp.asarray(obs), key), np.float64), config.max_speed)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
        safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
        h, _ = builder.build(env, {"u_flow": flow, "u_safe": safe}, config, cbf)
        if np.asarray(h).shape != (214,): raise RuntimeError("Incompatible true-t0 h schema")
        outputs.append(np.asarray(h, np.float32))
    return np.stack(outputs)


def model_free_predictions(h):
    selected = load(OUT / "generator_training.json")["selected"]
    model = genlib.Generator()
    tmp = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 214), jnp.float32), "Toy")
    p = serialization.from_bytes(tmp, Path(selected["checkpoint"]).read_bytes())
    norm = load(OLD / "dataset_manifest.json")["state_normalization"]["Toy"]
    x = (h - np.asarray(norm["mean"], np.float32)) / np.asarray(norm["std"], np.float32)
    raw = np.asarray(model.apply(p, jnp.asarray(x), "Toy"))
    mu, sig = map(np.asarray, genlib.dist_params(jnp.asarray(raw)))
    center = genlib.CENTER
    radius = genlib.RADIUS
    means = center + radius * np.tanh(mu)
    samples = []
    for i in range(len(h)):
        rng = np.random.default_rng(int(hashlib.sha256(f"mode_free_v1|seed{selected['seed']}|wide_ic|{i}".encode()).hexdigest()[:16], 16))
        z = rng.standard_normal((4, 3))
        samples.append(center + radius * np.tanh(mu[i] + sig[i] * z))
    return means, np.asarray(samples), sig


def prior_predictions(h):
    selector_model = prior.FrozenSelector()
    tmp = selector_model.init(jax.random.PRNGKey(0), jnp.zeros((1, 214)), jnp.zeros((1, 80)))
    selector_info = load(JOINT / "selected_models.json")["joint_100"]
    sp = serialization.from_bytes(tmp, Path(selector_info["checkpoint"]).read_bytes())
    norm = load(TOY / "normalization.json")
    x = (h - np.asarray(norm["h_mean"], np.float32)) / np.asarray(norm["h_std"], np.float32)
    logits, _ = selector_model.apply(sp, jnp.asarray(x), jnp.zeros((1, 80)))
    modes = np.argmax(np.asarray(logits), axis=1)
    selected = load(GENSRC / "selected_checkpoint.json")
    model, gp = prior.load_generator(selected)
    raw, _ = model.apply(gp, jnp.asarray(x), jnp.asarray(modes, jnp.int32), jnp.zeros((1, 80)), jnp.zeros((1,), jnp.int32))
    dz = np.asarray(prior.mean_delta(raw))
    anchors = prior.anchors()[0]
    az = (anchors - prior.AFF) / prior.SCALE
    arrays = np.load(DEFORM / "frozen_arrays.npz")
    eta = []
    for i, m in enumerate(modes):
        z = prior.project_domain(az[m] + dz[i], az[m], arrays["halfspace_A"], arrays["halfspace_b"])
        eta.append(prior.AFF + prior.SCALE * z)
    return anchors[modes], np.asarray(eta), modes


def critic_scores(h, samples):
    manifest = load(OLD / "dataset_manifest.json")
    sn = manifest["state_normalization"]["Toy"]
    ec = np.asarray(manifest["eta_normalization"]["center"], np.float32)
    es = np.asarray(manifest["eta_normalization"]["scale"], np.float32)
    x = (h - np.asarray(sn["mean"], np.float32)) / np.asarray(sn["std"], np.float32)
    model = ranklib.SingleCritic()
    logits = []
    used = []
    for seed in (17, 23, 41):
        template = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, 214)), jnp.zeros((1, 3)))
        cp = CRITIC / "models" / "secondary_combined" / "W1" / f"seed{seed}" / "checkpoint.msgpack"
        p = serialization.from_bytes(template, cp.read_bytes())
        hh = np.repeat(x, samples.shape[1], axis=0)
        zz = (samples.reshape(-1, 3) - ec) / es
        v = np.asarray(model.apply(p, jnp.asarray(hh), jnp.asarray(zz))).reshape(len(h), samples.shape[1])
        logits.append(v)
        used.append({"path": str(cp), "sha256": sha(cp)})
    scores = np.mean(np.stack(logits), axis=0)
    return scores, used


def main():
    wide = load(WIDE / "frozen_benchmark_manifest.json")
    episodes = wide["episodes"]
    if len(episodes) != 200: raise RuntimeError("Historical WIDE cohort changed")
    h = features_for_cohort(episodes, wide["environment"], wide["cbf"])
    means, samples, sigma = model_free_predictions(h)
    old_anchor, old_mean, modes = prior_predictions(h)
    scores, critics = critic_scores(h, samples)
    fixed = np.asarray(prior.anchors()[0][0], np.float64)
    rows = []
    for i, ep in enumerate(episodes):
        choices = {"fixed_common": fixed, "old_selector_anchor": old_anchor[i], "old_mode_generator_mean": old_mean[i], "generator_mean": means[i]}
        choices.update({f"sample_{k}": samples[i, k] for k in range(4)})
        pick = int(np.argmax(scores[i]))
        rows.append({"episode_index": i, "rollout_id": ep["rollout_id"], "initial_positions": ep["initial_positions"],
                     "source_group": f"wide_ic_frozen_{i:04d}", "h_sha256": hashlib.sha256(h[i].astype(np.float64).tobytes()).hexdigest(),
                     "old_mode_id_diagnostic_only": int(modes[i]), "eta": {name: [float(v) for v in eta] for name, eta in choices.items()},
                     "sigma": sigma[i].tolist(), "critic_scores": scores[i].tolist(), "critic_selected_sample": pick})
    payload = {"cohort": str(WIDE / "frozen_benchmark_manifest.json"), "cohort_sha256": sha(WIDE / "frozen_benchmark_manifest.json"),
               "generator_training_sha256": sha(OUT / "generator_training.json"), "critic_checkpoints": critics,
               "K": 4, "proposal_seed_rule": "SHA256(mode_free_v1|selected generator seed|wide_ic|episode_index)",
               "selection_before_outcomes": True, "states": rows}
    (OUT / "frozen_proposals.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(OUT / "cohort_features.npz", h_raw=h)
    print(json.dumps({"states": len(rows), "generator_mean_range": [means.min(0).tolist(), means.max(0).tolist()],
                      "sample_choice_counts": np.bincount(np.argmax(scores, axis=1), minlength=4).tolist()}, indent=2))


if __name__ == "__main__": main()
