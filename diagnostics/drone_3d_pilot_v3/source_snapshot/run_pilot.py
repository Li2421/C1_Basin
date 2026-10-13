"""Prepare -> shared preflight -> missing-only pilot -> journal -> shared merge.

One process, one CPU affinity, BLAS=1, no GPU or training. Keep source frozen
between prepare and run. A different source requires a different output folder.
"""
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, lookup_exact, transaction, uid
from . import pilot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_hashes():
    return {p.name: sha(p) for p in [Path(pilot.__file__), Path(__file__)]}


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "protocol.json").exists():
        raise ValueError("protocol already exists; use run for cached replay")
    config = asdict(pilot.Config())
    bounds, obstacles = pilot.geometry()
    scene = dict(name="Drone3D_RaisedSideAlcove_N2", config=config,
                 bounds=bounds.tolist(), obstacles=obstacles.tolist(), source=source_hashes())
    scid = uid("scn", scene)
    eta, eta_values, eta_hex = eta_identity([0., 0., 0.])
    controllers = {}
    for name in ["goal_cbf", "route_witness"]:
        payload = dict(scenario_uid=scid, controller=name, config=config, source=source_hashes(),
                       dynamics="3D single integrator, held velocity", rng="deterministic_after_seeded_initial_state_v1",
                       success="all goals + swept safety", deadlock="mutual_block_v1",
                       initial_controller_state="phase=-1,event_step=None,hold=None,step=0,empty_monitor_history",
                       checkpoint=None, eta_semantics="unused zero placeholder, no learned intervention")
        controllers[name] = dict(uid=uid("ctl", payload), payload=payload)
    states = []
    for seed in (17, 23, 41):
        physical = pilot.initial_state(seed)
        states.append(dict(seed=seed, physical=physical, uid=uid("state", dict(scenario=scid, physical=physical))))
    experiment = uid("exp", dict(path=str(folder.resolve()), scenario=scid))
    requests = [dict(scenario_uid=scid, state_uid=s["uid"], eta_uid=eta,
                     controller_uid=c["uid"], seed_keys=[canonical(dict(seed=s["seed"]))])
                for s in states for c in controllers.values()]
    protocol = dict(experiment_uid=experiment, scene=scene, scenario_uid=scid, eta_uid=eta,
                    states=states, controllers=controllers, source=source_hashes(), request_count=len(requests),
                    resource_cap=dict(cpu_affinity=1, blas_threads=1, gpu=0, workers=1,
                                      max_rollouts=6, max_steps_each=config["max_steps"]),
                    scope="engineering pilot, 3 symmetry-preserving initial states; no B15/Flow/Q claim")
    with connect(True) as db:
        historical_scenarios = [dict(r) for r in db.execute("SELECT name,scenario_uid FROM scenario ORDER BY name")]
    checked = ["shared_control/hard_projection.py", "shared_control/basis_families.py",
               "docs/part1_implementation_theory_audit_20261012.md", "bottleneck_family/README.md",
               "diagnostics/controller_loco_real_flow_v1/REPORT.md", "shared_rollout_db/README.md",
               "/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_v2/assets/toy_ckpt.pkl",
               "/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_v2/assets/db_ckpt.pkl"]
    write(folder / "history_audit.json", dict(
        checked=[dict(path=p, exists=Path(p).exists(), sha256=sha(p) if Path(p).is_file() else None) for p in checked],
        historical_scenarios=historical_scenarios,
        search="Repo docs/reports searched for quadrotor/quadcopter/drone/无人机/3D simulator; no matching 3D experiment found",
        reusable="2D projection objective, safety/goal semantics design, cache tooling, and negative control evidence",
        incompatible="2D states, N*2 action heads, finite wall segments, checkpoints and rollout labels do not execute this 3D geometry",
        missing="3D joint feasibility, actual mutually-blocked baseline, matched safe recovery, measured CPU cost",
        why_new="New 3D physical/action/obstacle semantics; not a critic-encoding change",
        no_training=True))
    write(folder / "protocol.json", protocol)
    write(folder / "planned_rollouts.json", dict(requests=requests))
    with connect() as db, transaction(db):
        db.execute("INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)",
                   (scid, scene["name"], uid("geometry", scene), canonical(scene)))
        db.execute("INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)",
                   (experiment, scene["name"], str(folder.resolve()), sha(folder / "protocol.json"), sha(__file__), canonical(protocol["resource_cap"])))
        db.execute("INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,normalized_eta_json,canonical_hex) VALUES(?,?,?,?,?,?)",
                   (eta, *eta_values, None, eta_hex))
        for s in states:
            db.execute("INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?,?)",
                       (s["uid"], scid, f"pilot_seed_{s['seed']}", uid("physical", s["physical"]), canonical(s["physical"]),
                        canonical(dict(step=0, monitor_history=[])), canonical(scene), canonical(dict(seed=s["seed"], path=str(folder.resolve()))), "CONTENT_EXACT"))
        for c in controllers.values():
            db.execute("INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (c["uid"], scid, None, uid("safety", scene), str(config["max_steps"]), str(config["dt"]),
                        "all_goals_swept_safety_mutual_block_v1", "no_eta_no_flow_v1", "deterministic_after_seeded_initial_state_v1", canonical(c["payload"]), "EXACT_PROFILE"))
    # Official shared preflight, before any simulator step.
    receipt = preflight(folder / "planned_rollouts.json")
    write(folder / "cache_preflight.json", receipt)
    print(json.dumps(receipt["summary"]))


def run(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    if protocol["source"] != source_hashes():
        raise ValueError("source changed after prepare; preserve this evidence and prepare a new folder")
    receipt = preflight(folder / "planned_rollouts.json")
    write(folder / "run_preflight.json", receipt)
    if receipt["summary"]["ambiguous"]:
        raise ValueError("ambiguous preflight")
    config = pilot.Config(**protocol["scene"]["config"])
    records = []; new = reused = 0; wall_start = time.perf_counter(); cpu_start = time.process_time()
    for s in protocol["states"]:
        for name, controller in protocol["controllers"].items():
            seed_key = canonical(dict(seed=s["seed"]))
            record_path = folder / f"seed{s['seed']}_{name}.json"
            trace_path = record_path.with_suffix(".npz")
            with connect(True) as db:
                cached = lookup_exact(db, s["uid"], protocol["eta_uid"], controller["uid"], [seed_key])
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if (record["state_uid"] != s["uid"] or record["controller_uid"] != controller["uid"]
                        or record["eta_uid"] != protocol["eta_uid"] or record["seed"] != s["seed"]
                        or not trace_path.exists() or sha(trace_path) != record["trace_sha256"]):
                    raise ValueError("cached artifact mismatch; restore original artifact, never silently rerun")
                if cached["missing_seeds"]:
                    append_journal([record], protocol["experiment_uid"], "cpu1_recovery")
                reused += 1
            elif cached["status"] != "INCOMPATIBLE":
                raise ValueError("compatible DB evidence exists; retrieve original trace rather than rerun")
            else:
                start = time.perf_counter()
                trace, result = pilot.rollout(s["physical"], name, config)
                identity = dict(state=s["uid"], eta=protocol["eta_uid"], controller=controller["uid"], seed=seed_key)
                record = dict(**result, cache_identity_schema="registered_exact_uid_v1",
                              scenario=protocol["scene"]["name"], scenario_uid=protocol["scenario_uid"],
                              state_uid=s["uid"], controller_uid=controller["uid"], eta_uid=protocol["eta_uid"],
                              rollout_uid=uid("roll", identity), eta=[0., 0., 0.], seed=s["seed"],
                              initial_positions=s["physical"]["positions"], initial_velocities=s["physical"]["velocities"],
                              goals=s["physical"]["goals"], flow_checkpoint_sha256=None,
                              timestamp=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start,
                              trace_path=str(trace_path.resolve()), source=source_hashes(),
                              initial_state_sha256=uid("initial", dict(physical=s["physical"], config=asdict(config), step=0, monitor_history=[])))
                np.savez_compressed(trace_path, **trace, metadata_json=np.array(json.dumps(record)))
                record["trace_sha256"] = sha(trace_path)
                write(record_path, record)
                append_journal([record], protocol["experiment_uid"], "cpu1")
                new += 1
            records.append(record)
            print(json.dumps({k: record[k] for k in ["seed", "controller", "outcome", "episode_length", "deadlock_event_step", "elapsed_seconds"]}), flush=True)
    pairs = []
    for seed in (17, 23, 41):
        a, b = [r for r in records if r["seed"] == seed]
        with np.load(a["trace_path"], allow_pickle=False) as ta, np.load(b["trace_path"], allow_pickle=False) as tb:
            prefix = np.array_equal(ta["positions"], tb["positions"][:len(ta["positions"])])
            spatial_rank = int(np.linalg.matrix_rank(tb["positions"][:, 0] - tb["positions"][0, 0], tol=1e-6))
        pairs.append(dict(seed=seed, baseline=a["outcome"], intervention=b["outcome"],
                          matched_full_initial_state=a["initial_state_sha256"] == b["initial_state_sha256"],
                          baseline_prefix_identical=prefix, recovery_path_spatial_rank=spatial_rank))
    proc_status = Path("/proc/self/status").read_text() if Path("/proc/self/status").exists() else ""
    hwm = next((float(line.split()[1])/1024 for line in proc_status.splitlines() if line.startswith("VmHWM:")), None)
    summary = dict(scope=protocol["scope"], new_rollouts=new, reused_rollouts=reused,
                   gpu_used=False, cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
                   run_wall_seconds=time.perf_counter()-wall_start, run_cpu_seconds=time.process_time()-cpu_start,
                   peak_rss_mib=hwm,
                   getrusage_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   memory_measurement="Linux /proc/self/status VmHWM; getrusage separately retained for inherited high-water accounting",
                   pairs=pairs, all_swept_safe=all(not r["collision"] and not r["numerical_failure"] for r in records),
                   minimum_wall_clearance_m=min(r["minimum_wall_clearance"] for r in records),
                   minimum_physical_pair_clearance_m=min(r["minimum_agent_clearance"] for r in records),
                   minimum_pair_ellipsoid_gap=min(r["minimum_pair_ellipsoid_gap"] for r in records),
                   experiment_uid=protocol["experiment_uid"])
    write(folder / ("summary.json" if new else "cache_replay_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, default=Path("diagnostics/drone_3d_pilot_v3"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args.output)
