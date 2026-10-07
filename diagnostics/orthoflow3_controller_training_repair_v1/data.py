"""Source-only true-t0 crossed supervision repair. Never opens target Q labels.

The three controllers, fixed eta panel and family split are frozen before new
outcomes. Workers append journals; only merge() writes outcomes to SQLite.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from diagnostics.orthoflow3_controller_information_probe_v1 import probe
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import intervention as old
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from new_benchmark_common import basin_dataset_v1 as bd
from new_benchmark_common.macflow import load_checkpoint
from new_benchmark_common.safety_eta3 import ScenarioRuntime
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import ROOT as DBROOT, canonical, connect, eta_identity, transaction, uid

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
EXPERIMENT = "exp_orthoflow3_controller_training_repair_v1"
SCENE = "ring_exchange"
read, write = probe.read, probe.write
CONTROLLERS = ("base", "alt", "second")
REVISION = "native_float32_v2"


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def validate_record(r, protocol, states_by_uid, pairs_by_key, controller_payloads):
    """Fail closed before insertion; exact identity includes execution semantics."""
    assert r["experiment_uid"] == EXPERIMENT
    assert r["execution_revision"] == REVISION
    assert r["flow_sampler_precision"] == "float32" and r["jax_enable_x64"] is False
    profile = next(c for c in protocol["profiles"] if c["controller_uid"] == r["controller_uid"])
    payload = controller_payloads[r["controller_uid"]]
    state = states_by_uid[r["state_uid"]]
    eu = eta_identity(r["eta"])[0]
    pair = pairs_by_key[(r["state_uid"], eu)]
    assert isinstance(r["future_index"], int) and 0 <= r["future_index"] < pair["target_seeds"]
    assert r["source_split"] == pair["split"] == state["split"]
    assert r["scenario"] == SCENE and r["controller_chain"] == "orthoflow3"
    assert r["flow_checkpoint_sha256"] == profile["sha256"] == payload["flow_checkpoint_sha256"]
    assert r["committed_t0_flow_sha256"] == protocol["base_flow_sha256"]
    for key in ("environment_sha256", "orthoflow3_sha256", "safety_projection_sha256"):
        assert r[key] == payload[key], key
    assert r["future_root_seed"] == payload["rng"]["future_root"]
    assert r["rng_namespace"] == bd.state_token(state["uid"])
    for a, b in (("initial_positions", "positions"), ("initial_velocities", "velocities"), ("goals", "goals")):
        assert canonical(r[a]) == canonical(state["physical"][b]), a
    assert all(r[k] in (False, True, 0, 1) for k in
               ("success", "deadlock", "timeout", "collision", "numerical_failure"))
    assert not (r["success"] and (r["collision"] or r["numerical_failure"]))
    assert 0 <= r["episode_length"] <= int(payload["horizon"])
    return eu


def prepare():
    jax.config.update("jax_enable_x64", False)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Source protocol already frozen; do not redesign using outcomes")
    pool = bd.parent_state_rows(SCENE)
    states = []
    for split, count in (("train", 24), ("validation", 6)):
        candidates = [s for s in pool if s["split"] == split]
        candidates.sort(key=lambda s: digest({"rule": "source_true_t0_repair_v1", "family": s["source_group"]}))
        assert len(candidates) >= count
        states.extend(copy.deepcopy(candidates[:count]))
    assert len({s["source_group"] for s in states}) == len(states)
    assert all(s["physical"]["timestep"] == 0 for s in states)
    held = read(probe.OUT / "held_controller_true_t0_v1/states.json")
    assert not ({s["uid"] for s in states} & {s["uid"] for s in held})
    assert not ({s["source_group"] for s in states} & {s["source_group"] for s in held})
    for i, s in enumerate(states):
        s["repair_index"] = i
    design = read(probe.OUT / "held_controller_robust_selection_v1/design.json")
    etas = list(zip(design["eta_uids"], design["eta_coordinates"]))
    assert len(etas) == 16 and all(eta_identity(e)[0] == u for u, e in etas)
    rt = bd.TrainingRuntime(SCENE, states, parent=True)
    base_uid = rt.controllers["orthoflow3"]["uid"]
    paths = [None,
             read(old.OUT / "balanced_expansion/protocol.json")["profiles"][SCENE]["alternate_path"],
             read(old.OUT / "second_variant/protocol.json")["profiles"][SCENE]["alternate_path"]]
    profiles = []
    with connect() as db, transaction(db):
        original = db.execute("SELECT * FROM controller_config WHERE controller_uid=?", (base_uid,)).fetchone()
        for name, path in zip(CONTROLLERS, paths):
            payload = json.loads(original["config_json"])
            sha = rt.checkpoint_sha if path is None else old.digest(path)
            if path:
                # Do not reuse stale copied controller_payload from historical protocols.
                payload.update(flow_checkpoint_sha256=sha,
                               committed_t0_flow_sha256=rt.checkpoint_sha,
                               conditioning=payload["conditioning"] + "+committed_base_t0_then_alt_future_v1",
                               intervention_runtime_sha256=old.digest(old.__file__))
            cid = uid("ctl", payload)
            db.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
                flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
                success_semantics_version,conditioning_version,rng_semantics_version,config_json,
                compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (cid, original["scenario_uid"], sha, original["orthoflow3_sha256"],
                 original["safety_config_hash"], original["horizon"], original["dt"],
                 original["success_semantics_version"], payload["conditioning"],
                 original["rng_semantics_version"], canonical(payload), "EXACT_PROFILE"))
            profiles.append({"name": name, "path": path, "sha256": sha, "controller_uid": cid})
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,
            code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)""",
            (EXPERIMENT, "source_true_t0_controller_supervision_repair", str(OUT),
             digest({"states": states, "etas": etas, "profiles": profiles}), old.digest(__file__),
             canonical({"target_labels": 0, "generator_modified": False})))
    pairs = [{"state_uid": s["uid"], "state_index": s["repair_index"], "split": s["split"],
              "eta_uid": eu, "eta": e, "eta_index": j, "target_seeds": 4 if s["split"] == "train" else 16}
             for s in states for j, (eu, e) in enumerate(etas)]
    requests = []
    for c in profiles:
        req = [{"state_uid": p["state_uid"], "eta_uid": p["eta_uid"], "controller_uid": c["controller_uid"],
                "seed_keys": [canonical({"future_index": i}) for i in range(p["target_seeds"])]} for p in pairs]
        write(OUT / f"planned_{c['name']}.json", {"requests": req})
        requests.extend(req)
    write(OUT / "states.json", states)
    write(OUT / "pairs.json", pairs)
    write(OUT / "planned_rollouts.json", {"requests": requests})
    write(OUT / "protocol.json", {
        "schema": "controller_training_repair_v1", "profiles": profiles,
        "base_controller_uid": base_uid, "base_flow_sha256": rt.checkpoint_sha,
        "environment_fingerprint": rt.manifest["environment_fingerprint"],
        "source_train_states": 24, "source_val_states": 6, "eta_per_state": 16,
        "pair_design": "complete state x shared exact eta x three compatible controllers",
        "state_selection": "sha256(source_true_t0_repair_v1,source_group), TRAIN and dev separately; outcomes not consulted",
        "eta_selection": "unaltered score-blind source-TRAIN panel frozen before held v11 outcomes",
        "source_eta_design_sha256": old.digest(probe.OUT / "held_controller_robust_selection_v1/design.json"),
        "source_target_overlap": 0, "source_train_val_family_overlap": 0,
        "train_seeds": 4, "validation_seeds": 16, "all_additional_compatible_history_retained": True,
        "weighting": "equal mean of per-controller observed Bernoulli NLL; matched pairs in every minibatch",
        "context": "unchanged H20 24 physical summaries; train-only normalization",
        "held_controller_labels_in_training": 0, "generator_changes": False,
        "safety_success_basis_horizon_unchanged": True,
        "source_requested_continuations": sum(len(r["seed_keys"]) for r in requests),
        "max_new_including_optional_unopened_confirmation": 15360,
        "held_stage1": "opened; regression only, not an independent confirmation",
        "held_stage2": "unopened 24 families; only after source gate and checkpoint freeze",
        "source_gate": {
            "nontrivial_certified_controller_reversals": ">=10 across >=3 validation families",
            "both_rankings_correct_fraction": ">=0.60 median across seeds",
            "full_vs_eta_only_and_additive": "positive net B15 rescue in at least 2 of 3 seeds",
            "correct_vs_wrong_context": "positive median selection rescue OR >=0.02 lower VAL NLL",
            "state_information": "state-dependent reversal accuracy >=0.60 median; shuffle must reduce accuracy",
            "failure_action": "diagnose source only; do not open independent target labels"},
        "runtime_sha256": old.digest(old.__file__), "code_sha256": old.digest(__file__),
    })
    write(OUT / "working_state.json", {"phase": "prepared", "new_rollout_executed": 0})
    print(json.dumps({"states": len(states), "pairs_per_controller": len(pairs), "requested": 9216}))


def physical():
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20
    if (OUT / "physical.json").exists():
        return
    states = read(OUT / "states.json")
    template = next(s["physical"] for s in read(old.OLD / "states.json") if s["scenario"] == SCENE)
    rt = h20.rc.RichRuntime(SCENE)
    values = []
    for s in states:
        p = copy.deepcopy(template)
        p.update(**{k: s["physical"][k] for k in ("positions", "velocities", "goals")},
                 remaining_fraction=1., remaining_seconds=template["max_seconds"], flow_committed=False)
        env = rt.core.reset(p)
        key = jax.random.fold_in(jax.random.PRNGKey(bd.CONDITIONING_FLOW_ROOT), bd.state_token(s["uid"]))
        p["flow"] = np.asarray(rt.base_flow(env, key), float).tolist()
        assert all(np.isfinite(v).all() for v in rep.entities(p).values())
        values.append({"state_uid": s["uid"], "physical": p})
    write(OUT / "physical.json", values)
    np.savez_compressed(OUT / "entities.npz", **rep.batch([rep.entities(v["physical"]) for v in values]))


def features(controller, shard, shards):
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20
    target = OUT / f"features_{controller}_{shard}of{shards}.json"
    if target.exists():
        return
    profile = next(c for c in read(OUT / "protocol.json")["profiles"] if c["name"] == controller)
    if profile["path"]:
        assert old.digest(profile["path"]) == profile["sha256"]
    h20.rc.OUT = OUT
    rt = h20.rc.RichRuntime(SCENE, profile["path"])
    states = read(OUT / "physical.json")
    result = []
    for i, p in enumerate(read(OUT / "pairs.json")):
        if i % shards != shard:
            continue
        s = states[p["state_index"]]
        assert s["state_uid"] == p["state_uid"]
        value = h20.rc.cached(rt, s, p["eta"])
        result.append({"pair_index": i, "valid": value["valid"], "error": value.get("error"),
                       "context": value["features"]["mean"] if value["valid"] else None})
        if len(result) % 32 == 0:
            print(json.dumps({"controller": controller, "features": len(result)}), flush=True)
    write(target, result)


def run(controller, shard, shards):
    # Historical context imports changed process-global precision. Keep them
    # off the native task executor path. Pin native Flow/RNG precision
    # explicitly before checkpoint initialization and keep physical NumPy float64.
    jax.config.update("jax_enable_x64", False)
    assert jax.random.normal(jax.random.PRNGKey(123), (1,)).dtype == np.dtype("float32")
    protocol = read(OUT / "protocol.json")
    assert old.digest(old.__file__) == protocol["runtime_sha256"]
    profile = next(c for c in protocol["profiles"] if c["name"] == controller)
    pre = read(OUT / f"cache_preflight_{controller}.json")
    missing = {(r["state_uid"], r["eta_uid"], sk) for r in pre["details"]
               if r["status"] != "AGGREGATE_REUSE" for sk in r["missing_seeds"]}
    states = read(OUT / "states.json")
    jobs = [(p, i) for p in read(OUT / "pairs.json") for i in range(p["target_seeds"])
            if (p["state_uid"], p["eta_uid"], canonical({"future_index": i})) in missing]
    assert len(jobs) == pre["summary"]["genuinely_missing"]
    jobs = [j for i, j in enumerate(jobs) if i % shards == shard]
    rt = bd.TrainingRuntime(SCENE, states, parent=True)
    assert rt.controllers["orthoflow3"]["uid"] == protocol["base_controller_uid"]
    assert rt.checkpoint_sha == protocol["base_flow_sha256"]
    counter = {"calls": 0}
    if profile["path"]:
        assert old.digest(profile["path"]) == profile["sha256"]
        alt, _ = load_checkpoint(profile["path"], expected_environment_fingerprint=protocol["environment_fingerprint"])
        base = rt.agent
        def flow(env, key):
            rt.agent = base if counter["calls"] == 0 else alt
            counter["calls"] += 1
            return ScenarioRuntime.flow_world(rt, env, key)
        rt.flow_world = flow
    # Resume a worker from its own immutable journals; never repeat numerical failures.
    done = set()
    for path in (DBROOT / "journals" / EXPERIMENT).glob(f"{REVISION}_{controller}_shard{shard}_*.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)["record"]
            done.add((r["state_uid"], eta_identity(r["eta"])[0], r["future_index"]))
    completed = 0
    for p, seed in jobs:
        if (p["state_uid"], p["eta_uid"], seed) in done:
            continue
        counter["calls"] = 0
        assert not jax.config.x64_enabled, "Native Ring Flow sampler must remain float32"
        result = rt.rollout(states[p["state_index"]], np.asarray(p["eta"], float), seed, "orthoflow3")
        result.update(controller_uid=profile["controller_uid"], experiment_uid=EXPERIMENT,
                      flow_checkpoint_sha256=profile["sha256"], source_split=p["split"],
                      committed_t0_flow_sha256=protocol["base_flow_sha256"],
                      flow_sampler_precision="float32", jax_enable_x64=False, execution_revision=REVISION)
        assert result["state_uid"] == p["state_uid"] and eta_identity(result["eta"])[0] == p["eta_uid"]
        append_journal([{"schema": "controller_training_repair_v1", "record": result}],
                       EXPERIMENT, f"{REVISION}_{controller}_shard{shard}")
        completed += 1
        if completed % 32 == 0:
            print(json.dumps({"controller": controller, "shard": shard, "done": completed, "planned": len(jobs)}), flush=True)
    print(json.dumps({"controller": controller, "shard": shard, "complete": completed}), flush=True)


def merge():
    protocol = read(OUT / "protocol.json")
    states_by_uid = {s["uid"]: s for s in read(OUT / "states.json")}
    pairs_by_key = {(p["state_uid"], p["eta_uid"]): p for p in read(OUT / "pairs.json")}
    allowed = {(p["state_uid"], p["eta_uid"], c["controller_uid"])
               for p in read(OUT / "pairs.json") for c in protocol["profiles"]}
    stats = Counter()
    with connect() as db:
        controller_payloads = {c["controller_uid"]: json.loads(db.execute(
            "SELECT config_json FROM controller_config WHERE controller_uid=?",
            (c["controller_uid"],)).fetchone()[0]) for c in protocol["profiles"]}
        for path in sorted((DBROOT / "journals" / EXPERIMENT).glob(f"{REVISION}_*.jsonl")):
            source = uid("src", {"path": str(path.resolve())})
            if db.execute("SELECT 1 FROM source_file WHERE source_uid=?", (source,)).fetchone():
                stats["already_merged_journals"] += 1
                continue
            with transaction(db):
                lines = path.read_text().splitlines()
                journal_counts = Counter()
                db.execute("INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)",
                           (source, EXPERIMENT, str(path), old.digest(path), ".jsonl", "SEED_EXACT", len(lines)))
                for number, line in enumerate(lines, 1):
                    envelope = json.loads(line)
                    assert envelope["schema"] == "controller_training_repair_v1"
                    r = envelope["record"]
                    eu = validate_record(r, protocol, states_by_uid, pairs_by_key, controller_payloads)
                    assert (r["state_uid"], eu, r["controller_uid"]) in allowed
                    sk = canonical({"future_index": r["future_index"]})
                    rid = uid("roll", {"state": r["state_uid"], "eta": eu, "controller": r["controller_uid"], "seed": sk})
                    core = tuple(int(r[k]) for k in ("success", "deadlock", "timeout", "collision", "numerical_failure"))
                    previous = db.execute("SELECT * FROM rollout WHERE rollout_uid=?", (rid,)).fetchone()
                    if previous:
                        if core != tuple(previous[k] for k in ("success", "deadlock", "timeout", "collision", "numerical_failure")):
                            db.execute("UPDATE rollout SET conflict_quarantined=1 WHERE rollout_uid=?", (rid,))
                            db.execute("""INSERT OR IGNORE INTO conflict(conflict_uid,entity_type,identity_key,
                                existing_json,incoming_json,source_file) VALUES(?,?,?,?,?,?)""",
                                (uid("conflict", {"rollout_uid": rid, "incoming": digest(r)}), "rollout", rid,
                                 canonical(dict(previous)), canonical(r), str(path)))
                            stats["conflicts_quarantined"] += 1
                            journal_counts["ambiguous"] += 1
                            continue
                        stats["duplicate"] += 1
                        journal_counts["duplicate"] += 1
                    else:
                        db.execute("""INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,
                            continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,
                            episode_length,j_def,min_wall_distance,min_agent_distance,outcome,experiment_uid,
                            original_source_file,timestamp,compatibility_quality,raw_record_hash)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (rid, r["state_uid"], eu, r["controller_uid"], sk, sk, *core,
                             r.get("episode_length"), r.get("J_def"), r.get("minimum_wall_clearance"),
                             r.get("minimum_agent_clearance"), r.get("outcome"), EXPERIMENT,
                             str(path), r.get("timestamp"), "EXACT_REUSE", digest(r)))
                        stats["inserted"] += 1
                        journal_counts["imported"] += 1
                        stats["numerical_failure"] += core[-1]
                    db.execute("INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)", (rid, source, number))
                db.execute("UPDATE source_file SET rows_imported=?,rows_duplicate=?,rows_ambiguous=? WHERE source_uid=?",
                           (journal_counts["imported"], journal_counts["duplicate"], journal_counts["ambiguous"], source))
            stats["journals"] += 1
        db.execute("UPDATE experiment SET new_rollout_count=(SELECT COUNT(*) FROM rollout WHERE experiment_uid=?) WHERE experiment_uid=?", (EXPERIMENT, EXPERIMENT))
        preflight = read(OUT / "cache_preflight.json")["summary"]
        db.execute("UPDATE experiment SET reused_rollout_count=? WHERE experiment_uid=?",
                   (preflight["exact_reusable"], EXPERIMENT))
        stats["total_new_records_in_db"] = db.execute("SELECT COUNT(*) FROM rollout WHERE experiment_uid=?", (EXPERIMENT,)).fetchone()[0]
        stats["total_numerical_records_in_db"] = db.execute("SELECT COUNT(*) FROM rollout WHERE experiment_uid=? AND numerical_failure=1", (EXPERIMENT,)).fetchone()[0]
        db.commit()
    write(OUT / "merge_audit.json", dict(stats))
    print(json.dumps(dict(stats)))
    if stats["conflicts_quarantined"]:
        raise RuntimeError("Outcome conflict quarantined and committed; audit before continuing")
    from .alignment_audit import main as audit_alignment
    audit_alignment()


def materialize(shards=2):
    protocol = read(OUT / "protocol.json")
    pairs = read(OUT / "pairs.json")
    n = len(pairs)
    success, failure, numerical = [np.zeros((3, n), np.float32) for _ in range(3)]
    standard_success, standard_failure = [np.zeros((3, n), np.float32) for _ in range(2)]
    context = np.full((3, n, 24), np.nan, np.float32)
    provenance = []
    with connect(True) as db:
        for ci, c in enumerate(protocol["profiles"]):
            for sh in range(shards):
                for r in read(OUT / f"features_{c['name']}_{sh}of{shards}.json"):
                    if r["valid"]:
                        context[ci, r["pair_index"]] = r["context"]
            for i, p in enumerate(pairs):
                rows = db.execute("""SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?
                    AND conflict_quarantined=0 AND compatibility_quality='EXACT_REUSE' ORDER BY seed_key""",
                    (p["state_uid"], p["eta_uid"], c["controller_uid"])).fetchall()
                good = [r for r in rows if not r["numerical_failure"]]
                success[ci, i] = sum(r["success"] for r in good)
                failure[ci, i] = len(good) - success[ci, i]
                numerical[ci, i] = sum(r["numerical_failure"] for r in rows)
                standard = [r for r in good if r["seed_key"] in
                            {canonical({"future_index": seed}) for seed in range(16)}]
                standard_success[ci, i] = sum(r["success"] for r in standard)
                standard_failure[ci, i] = len(standard)-standard_success[ci, i]
                provenance.append({"controller": c["name"], "pair_index": i,
                                   "rollout_uids": [r["rollout_uid"] for r in rows]})
                if not good:
                    aggregates = db.execute("""SELECT * FROM aggregate_evidence WHERE state_uid=? AND eta_uid=?
                        AND controller_uid=? AND conflict_quarantined=0 ORDER BY n_trials DESC""",
                        (p["state_uid"], p["eta_uid"], c["controller_uid"])).fetchall()
                    if aggregates:
                        a = aggregates[0]
                        success[ci, i], failure[ci, i] = a["n_success"], a["n_trials"]-a["n_success"]
                        provenance[-1]["aggregate_uid"] = a["aggregate_uid"]
    valid = np.isfinite(context).all(-1) & ((success + failure) > 0)
    np.savez_compressed(OUT / "dataset.npz", success=success, failure=failure, numerical=numerical,
                        standard_success=standard_success, standard_failure=standard_failure,
                        context=context, valid=valid,
                        eta=np.asarray([p["eta"] for p in pairs], np.float32),
                        state_index=np.asarray([p["state_index"] for p in pairs]),
                        split=np.asarray([p["split"] for p in pairs]))
    write(OUT / "dataset_provenance.json", provenance)
    audit = {"canonical_pairs_per_controller": n, "valid": valid.sum(axis=1).tolist(),
             "observed_trials": (success + failure).sum(axis=1).tolist(),
             "success": success.sum(axis=1).tolist(), "failure": failure.sum(axis=1).tolist(),
             "numerical_seeds_separate": numerical.sum(axis=1).tolist(),
             "target_labels": 0, "source_train_val_overlap": 0}
    audit["train_weight_audit"] = []
    tr = np.asarray([p["split"] == "train" for p in pairs])
    for ci, c in enumerate(protocol["profiles"]):
        trials = (success[ci]+failure[ci])*valid[ci]*tr
        total = float(trials.sum())
        q = success[ci]/np.maximum(success[ci]+failure[ci], 1.)
        weights = [float(sum(trials[i] for i, p in enumerate(pairs) if p["state_index"] == j))
                   for j in sorted({p["state_index"] for p in pairs if p["split"] == "train"})]
        audit["train_weight_audit"].append({"controller": c["name"], "observed_training_trials": total,
            "objective_controller_mass": 1/3,
            "max_over_median_state_trial_weight": max(weights)/max(1., float(np.median(weights))),
            "effective_trial_mass_by_empirical_Q": {
                "Q_zero": float(trials[q == 0].sum()/max(1., total)),
                "intermediate": float(trials[(q > 0) & (q < 1)].sum()/max(1., total)),
                "Q_one": float(trials[q == 1].sum()/max(1., total))},
            "note": "Q=1 with four trials is not a B15 certification"})
    write(OUT / "data_audit.json", audit)
    print(json.dumps(audit))


def postflight_audit():
    requests = read(OUT / "planned_rollouts.json")["requests"]
    counts = Counter()
    missing = []
    with connect(True) as db:
        for req in requests:
            records = {r["seed_key"]: r for r in db.execute("""SELECT * FROM rollout
                WHERE state_uid=? AND eta_uid=? AND controller_uid=?""",
                (req["state_uid"], req["eta_uid"], req["controller_uid"]))}
            for key in req["seed_keys"]:
                row = records.get(key)
                if row is None:
                    missing.append({"state_uid": req["state_uid"], "eta_uid": req["eta_uid"], "controller_uid": req["controller_uid"], "seed_key": key})
                elif row["conflict_quarantined"]:
                    counts["quarantined"] += 1
                elif row["numerical_failure"]:
                    counts["numerical_not_imputed"] += 1
                else:
                    counts["exact_reusable"] += 1
                if row:
                    counts["collision"] += row["collision"]
    result = {"requested": 9216, **dict(counts), "not_attempted": missing,
              "all_attempted_records_in_db": len(missing) == 0,
              "no_numerical_seed_rerun_or_label_imputation": True}
    write(OUT / "postflight_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "not_attempted"}))
    assert not missing and not counts["quarantined"]
    from .alignment_audit import main as audit_alignment
    audit_alignment("all")
    from .evidence_audit import main as audit_evidence
    audit_evidence()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "physical", "features", "run", "merge", "materialize", "postflight_audit"))
    p.add_argument("--controller", choices=CONTROLLERS, default="base")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=2)
    a = p.parse_args()
    if a.action in ("features", "run"):
        globals()[a.action](a.controller, a.shard, a.shards)
    elif a.action == "materialize":
        materialize(a.shards)
    else:
        globals()[a.action]()
