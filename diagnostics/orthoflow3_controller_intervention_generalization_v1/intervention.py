"""Matched future-Flow interventions on source-family training states.

The canonical first action is preserved.  Only the future Flow checkpoint is
changed; plant, OrthoFlow3, safety, horizon, outcome and RNG semantics remain
those of the frozen source runtime.  Explicit controller identities prevent
accidental reuse of canonical rollout records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "diagnostics/orthoflow3_loso_partial_count_v1"
SCENES = ("double_bottleneck", "four_way_intersection", "ring_exchange")
ALT = {
    "double_bottleneck": ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_primary_75pct.pkl",
    "four_way_intersection": ROOT / "diagnostics/four_way_intersection_stage1/base_u_v12_world_timeout_goal_recovery_canonical_source_balanced_macflow/best.pkl",
    "ring_exchange": ROOT / "diagnostics/ring_exchange_stage1/base_u_v9_local_macflow/best.pkl",
}
EXPERIMENT = "exp_orthoflow3_controller_intervention_generalization_v1"


def read(path):
    return json.loads(Path(path).read_text())


def write(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(value):
    return hashlib.sha256(("intervention-v1|" + str(value)).encode()).hexdigest()


def prepare():
    from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, transaction, uid

    if (OUT / "protocol.json").exists():
        print("frozen protocol already exists")
        return
    rows = pq.read_table(OLD / "all_pairs.parquet").to_pylist()
    states = read(OLD / "states.json")
    by_state = defaultdict(list)
    for row in rows:
        if row["scenario"] in SCENES and row["full_standard_16"]:
            by_state[row["state_index"]].append(row)

    # Availability and source-family ranks select states; outcome can choose
    # informative TRAIN eta only. Validation eta are selected without labels.
    selected = []
    for scene in SCENES:
        for split, limit in (("train", 4), ("validation", 2)):
            possible = [(i, states[i]) for i in by_state if states[i]["scenario"] == scene
                        and states[i]["split"] == split and len(by_state[i]) >= 4]
            possible.sort(key=lambda z: rank(z[1]["family"]))
            assert len(possible) >= limit, (scene, split, len(possible))
            for i, state in possible[:limit]:
                options = sorted(by_state[i], key=lambda r: rank(r["eta_uid"]))
                if split == "train":
                    pos = [r for r in options if r["b15_confirmed"]]
                    neg = [r for r in options if r["non_b15_confirmed"]]
                    pick = pos[:2] + neg[:2]
                    pick += [r for r in options if r not in pick][:4 - len(pick)]
                else:
                    pick = options[:4]
                assert len(pick) == 4
                selected.extend({"scene": scene, "split": split, "state_index": i,
                                 "state_uid": state["state_uid"], "family": state["family"],
                                 "eta": r["eta"], "eta_uid": r["eta_uid"],
                                 "base_controller_uid": r["controller_uid"],
                                 "base_success": r["standard_success"],
                                 "base_failure": r["standard_failure"]} for r in pick)

    profiles = {}
    with connect() as db, transaction(db):
        for scene in SCENES:
            base_ids = {r["base_controller_uid"] for r in selected if r["scene"] == scene}
            assert len(base_ids) == 1, (scene, base_ids)
            base_uid = next(iter(base_ids))
            base = dict(db.execute("SELECT * FROM controller_config WHERE controller_uid=?", (base_uid,)).fetchone())
            assert base["compatibility_quality"] == "EXACT_PROFILE"
            config = json.loads(base["config_json"])
            alt_hash = digest(ALT[scene])
            canonical_hash = base["flow_checkpoint_sha256"]
            assert alt_hash != canonical_hash
            with ALT[scene].open("rb") as handle:
                ckpt = pickle.load(handle)
            if scene == "double_bottleneck":
                from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old
                expected = pickle.load(old.CHECKPOINT.open("rb"))["config"]["environment_fingerprint"]
            else:
                from new_benchmark_common.safety_eta3 import SCENARIOS
                expected = read(SCENARIOS[scene]["dataset"] / "manifest.json")["environment_fingerprint"]
            assert ckpt["config"]["environment_fingerprint"] == expected
            config["flow_checkpoint_sha256"] = alt_hash
            config["committed_t0_flow_sha256"] = canonical_hash
            config["conditioning"] = str(config.get("conditioning", "")) + "+committed_base_t0_then_alt_future_v1"
            config["intervention_runtime_sha256"] = digest(__file__)
            alt_uid = uid("ctl", config)
            profiles[scene] = {"base_controller_uid": base_uid, "alternate_controller_uid": alt_uid,
                               "base_flow_sha256": canonical_hash, "alternate_flow_sha256": alt_hash,
                               "alternate_path": str(ALT[scene]), "environment_fingerprint": expected,
                               "controller_payload": config}
            db.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
                flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
                success_semantics_version,conditioning_version,rng_semantics_version,config_json,
                compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (alt_uid, base["scenario_uid"], alt_hash, base["orthoflow3_sha256"],
                 base["safety_config_hash"], base["horizon"], base["dt"],
                 base["success_semantics_version"], config["conditioning"],
                 base["rng_semantics_version"], canonical(config), "EXACT_PROFILE"))
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,
            code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)""",
            (EXPERIMENT, "controller_intervention_generalization_v1", str(OUT),
             digest(__file__), digest(__file__), canonical({"source_only": True, "pilot": True})))
    requests = []
    for row in selected:
        row["alternate_controller_uid"] = profiles[row["scene"]]["alternate_controller_uid"]
        for controller in (row["base_controller_uid"], row["alternate_controller_uid"]):
            requests.append({"state_uid": row["state_uid"], "eta_uid": row["eta_uid"],
                             "controller_uid": controller,
                             "seed_keys": [canonical({"future_index": i}) for i in range(16)]})
    assert len(requests) == 144
    protocol = {"schema": "matched_future_flow_intervention_v1", "experiment_uid": EXPERIMENT,
                "source_only": True, "frozen_loso_target_states_used": False,
                "intervention": "canonical Flow at first step, alternate compatible frozen Flow thereafter",
                "safety_or_success_changed": False, "basis_changed": False,
                "state_selection": "SHA256-ranked source families with >=4 existing Q16 pairs",
                "train_eta_selection": "up to two B15 and two non-B15 existing source-TRAIN Q16 pairs",
                "validation_eta_selection": "hash rank only; validation outcomes not used",
                "states_per_scene": {"train": 4, "validation": 2}, "eta_per_state": 4,
                "standard_seeds": list(range(16)), "profiles": profiles,
                "source_pairs_sha256": digest(OLD / "all_pairs.parquet"),
                "code_sha256": digest(__file__),
                "pilot_criterion": "controller Q contrast, B15 flips and label integrity before expansion",
                "note": "This pilot is not itself LOSO evidence; never train on a held-out target scene."}
    write("protocol.json", protocol)
    write("pairs.json", selected)
    write("planned_rollouts.json", {"requests": requests})
    print(json.dumps({"states": 18, "pairs": len(selected), "requested": len(requests) * 16,
                      "source_scenes": list(SCENES)}))


def _runtime(scene, profile):
    import jax
    from new_benchmark_common import basin_dataset_v1 as bd

    if scene == "double_bottleneck":
        states = bd.double_runtime_states()
        relevant = [x for x in states if x["uid"] in {r["state_uid"] for r in read(OUT / "pairs.json") if r["scene"] == scene}]
        rt = bd.DoubleTrainingRuntime(relevant, profile["base_controller_uid"])
        from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old
        policy, _ = old.load_checkpoint(Path(profile["alternate_path"]), next(iter(rt.datasets.values())).environment_fingerprint)
        base = rt.policy

        class Switch:
            def __init__(self): self.calls = 0
            def sample_actions(self, observation, key):
                use = base if self.calls == 0 else policy
                self.calls += 1
                return use.sample_actions(observation, key)

        switch = Switch()
        rt.policy = switch
        return rt, {s["uid"]: s for s in relevant}, lambda: setattr(switch, "calls", 0)
    else:
        from new_benchmark_common.macflow import load_checkpoint
        from new_benchmark_common.safety_eta3 import SCENARIOS, ScenarioRuntime
        rt = bd.TrainingRuntime(scene, [], parent=False)
        assert rt.checkpoint_sha == profile["base_flow_sha256"]
        alternate, _ = load_checkpoint(Path(profile["alternate_path"]),
                                       expected_environment_fingerprint=profile["environment_fingerprint"])
        base = rt.agent
        counter = {"calls": 0}

        def flow(env, key):
            rt.agent = base if counter["calls"] == 0 else alternate
            counter["calls"] += 1
            return ScenarioRuntime.flow_world(rt, env, key)

        rt.flow_world = flow
        states = read(OLD / "states.json")
        indices = {r["state_index"] for r in read(OUT / "pairs.json") if r["scene"] == scene}
        by = {}
        for i in indices:
            s = states[i]
            p = s["physical"]
            timestep = int(round((p["max_seconds"] - p["remaining_seconds"]) / p["dt"]))
            by[s["state_uid"]] = {"uid": s["state_uid"], "alias": s["family"], "index": i,
                                  "source_group": s["family"],
                                  "physical": {"positions": p["positions"], "velocities": p["velocities"],
                                               "goals": p["goals"], "timestep": timestep}}
        return rt, by, lambda: counter.__setitem__("calls", 0)


def run(scene, shard, shards):
    from shared_rollout_db.src.cache_writer import append_journal
    from shared_rollout_db.src.rollout_db import canonical, eta_identity

    protocol = read(OUT / "protocol.json")
    assert digest(__file__) == protocol["code_sha256"]
    profile = protocol["profiles"][scene]
    assert digest(profile["alternate_path"]) == profile["alternate_flow_sha256"]
    pre = read(OUT / "cache_preflight.json")
    missing = {(r["state_uid"], r["eta_uid"], r["controller_uid"], seed)
               for r in pre["details"] for seed in r["missing_seeds"]}
    jobs = [(row, seed) for row in read(OUT / "pairs.json") if row["scene"] == scene
            for seed in range(16)
            if (row["state_uid"], row["eta_uid"], row["alternate_controller_uid"],
                canonical({"future_index": seed})) in missing]
    jobs = [job for i, job in enumerate(jobs) if i % shards == shard]
    rt, states, reset = _runtime(scene, profile)
    completed = 0
    for row, seed in jobs:
        reset()
        result = rt.rollout(states[row["state_uid"]], np.asarray(row["eta"], float), seed, "orthoflow3")
        result["controller_uid"] = row["alternate_controller_uid"]
        result["base_controller_uid"] = row["base_controller_uid"]
        result["intervention"] = protocol["intervention"]
        result["flow_checkpoint_sha256"] = profile["alternate_flow_sha256"]
        result["committed_t0_flow_sha256"] = profile["base_flow_sha256"]
        assert result["state_uid"] == row["state_uid"]
        assert eta_identity(result["eta"])[0] == row["eta_uid"]
        append_journal([{"schema": "matched_future_flow_intervention_v1", "record": result}],
                       EXPERIMENT, f"{scene}_shard{shard}")
        completed += 1
        if completed % 16 == 0:
            print(json.dumps({"scene": scene, "shard": shard, "completed": completed}), flush=True)
    print(json.dumps({"scene": scene, "shard": shard, "complete": completed}), flush=True)


def merge():
    from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, transaction, uid

    protocol = read(OUT / "protocol.json")
    allowed = {(r["state_uid"], r["eta_uid"], r["alternate_controller_uid"])
               for r in read(OUT / "pairs.json")}
    stats = Counter()
    journal = ROOT / "shared_rollout_db/journals" / EXPERIMENT
    with connect() as db:
        for path in sorted(journal.glob("*.jsonl")):
            lines = path.read_text().splitlines()
            source_uid = uid("src", {"path": str(path.resolve())})
            with transaction(db):
                db.execute("""INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,
                    file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)""",
                    (source_uid, EXPERIMENT, str(path.resolve()), digest(path), ".jsonl", "SEED_EXACT", len(lines)))
                for number, line in enumerate(lines, 1):
                    envelope = json.loads(line)
                    assert envelope["schema"] == "matched_future_flow_intervention_v1"
                    row = envelope["record"]
                    eta_uid = eta_identity(row["eta"])[0]
                    key = (row["state_uid"], eta_uid, row["controller_uid"])
                    assert key in allowed and 0 <= row["future_index"] < 16
                    sk = canonical({"future_index": row["future_index"]})
                    rid = uid("roll", {"state": row["state_uid"], "eta": eta_uid,
                                       "controller": row["controller_uid"], "seed": sk})
                    core = tuple(int(row[k]) for k in ("success", "deadlock", "timeout", "collision", "numerical_failure"))
                    old = db.execute("SELECT * FROM rollout WHERE rollout_uid=?", (rid,)).fetchone()
                    if old:
                        previous = tuple(old[k] for k in ("success", "deadlock", "timeout", "collision", "numerical_failure"))
                        if core != previous:
                            db.execute("UPDATE rollout SET conflict_quarantined=1 WHERE rollout_uid=?", (rid,))
                            stats["conflicts"] += 1
                            continue
                        stats["duplicates"] += 1
                    else:
                        db.execute("""INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,
                            continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,
                            episode_length,j_def,min_wall_distance,min_agent_distance,outcome,experiment_uid,
                            original_source_file,timestamp,compatibility_quality,raw_record_hash)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (rid, row["state_uid"], eta_uid, row["controller_uid"], sk, sk, *core,
                             row.get("episode_length"), row.get("J_def"), row.get("minimum_wall_clearance"),
                             row.get("minimum_agent_clearance"), row.get("outcome"), EXPERIMENT,
                             str(path), row.get("timestamp"), "EXACT_REUSE",
                             hashlib.sha256(canonical(row).encode()).hexdigest()))
                        stats["inserted"] += 1
                    db.execute("INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)", (rid, source_uid, number))
            stats["journals"] += 1
        db.execute("UPDATE experiment SET new_rollout_count=?,end_time=CURRENT_TIMESTAMP WHERE experiment_uid=?",
                   (stats["inserted"], EXPERIMENT))
        db.commit()
    write("merge_audit.json", dict(stats))
    print(json.dumps(dict(stats)))


def analyze():
    from shared_rollout_db.src.rollout_db import connect

    pairs = read(OUT / "pairs.json")
    results = []
    with connect(True) as db:
        for row in pairs:
            hits = db.execute("""SELECT success,numerical_failure FROM rollout WHERE state_uid=? AND eta_uid=?
                AND controller_uid=? AND conflict_quarantined=0 AND compatibility_quality='EXACT_REUSE'""",
                (row["state_uid"], row["eta_uid"], row["alternate_controller_uid"])).fetchall()
            valid = [x for x in hits if not x["numerical_failure"]]
            s = sum(x["success"] for x in valid)
            results.append({**row, "alt_valid": len(valid), "alt_success": s,
                            "alt_Q16": s / 16, "base_Q16": row["base_success"] / 16,
                            "delta_Q16": (s - row["base_success"]) / 16,
                            "alt_B15": len(valid) == 16 and s >= 15,
                            "base_B15": row["base_success"] >= 15})
    with (OUT / "intervention_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    summary = {"pairs": len(results), "new_continuations": len(results) * 16,
               "by_scene": {scene: {"pairs": sum(x["scene"] == scene for x in results),
                                     "large_delta_Q": sum(x["scene"] == scene and abs(x["delta_Q16"]) >= .25 for x in results),
                                     "B15_flip": sum(x["scene"] == scene and x["base_B15"] != x["alt_B15"] for x in results),
                                     "numerical_unresolved": sum(x["scene"] == scene and x["alt_valid"] < 16 for x in results)}
                            for scene in SCENES}}
    write("pilot_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "merge", "analyze"))
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run(args.scene, args.shard, args.shards)
    elif args.action == "merge": merge()
    else: analyze()
