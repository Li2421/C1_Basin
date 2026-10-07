"""Run one frozen η's matched Ring continuations into this isolated data directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
MAIN_ROOT = HERE.parents[1]
POC_ROOT = MAIN_ROOT.parent / "Basin_C1_flow_field_poc_20261004"
sys.path[:0] = [str(POC_ROOT), str(MAIN_ROOT)]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import jax
import numpy as np
from poc.field_controller import sha
from validation_stage2.run_rollouts import make_runtime, serial
from validation_stage2.factorial_controller import action
from family_study_v1.runner import CONFIG, key_for, controller_uid

CHAINS = ("TT", "TF", "FT", "FF")


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, default=serial, allow_nan=False) + "\n")
    tmp.replace(path)


def run_one(rt, state, eta_index, eta, seed, chain):
    case = {"id": hashlib.sha256(json.dumps([state["uid"], eta.tolist(), seed], separators=(",", ":")).encode()).hexdigest()[:20],
            "state": state, "eta_name": f"illustrative_sobol_{eta_index:02d}", "eta": eta.tolist(), "seed": seed}
    env = rt.make_env(state)
    eta_arr = np.asarray(eta, dtype=np.float64)
    error = None
    term = "timeout"
    steps = 0
    calls = 0
    max_violation = 0.
    max_pre = 0.
    t0 = time.monotonic()
    for step in range(rt.config.max_steps):
        try:
            a, cert, n, extra = action(rt, env, key_for(case, step), eta_arr, chain)
            calls += n
            max_violation = max(max_violation, float(cert["max_violation"]))
            if extra["pre_terminal_violation"] is not None:
                max_pre = max(max_pre, float(extra["pre_terminal_violation"]))
            _, _, done, info = env.step(a)
            term = info["termination"]
            steps += 1
            if done:
                break
        except Exception as exc:
            error = {"step": step, "type": type(exc).__name__, "message": str(exc)[:2400]}
            term = "numerical_failure"
            break
    summary = env.summary()
    collision = any(summary.get(k, False) for k in ("wall_collision", "obstacle_collision", "outer_collision", "agent_collision"))
    return {"case_id": case["id"], "state_uid": state["uid"], "scenario": "ring_exchange", "eta_index": eta_index,
            "eta": eta.tolist(), "seed": seed, "chain": chain, "controller_uid": controller_uid(rt, chain),
            "success": bool(summary["collision_free_success"] and term == "success" and error is None),
            "collision": bool(collision), "termination": term, "error": error, "steps": steps,
            "terminal_goal_error": float(np.linalg.norm(env.goals - env.positions, axis=1).sum()),
            "max_action_violation": max_violation, "max_pre_terminal_violation": max_pre,
            "projection_calls": calls, "elapsed_seconds": time.monotonic() - t0, "plant_summary": summary,
            "protocol_sha256": json.loads((HERE / "manifest.json").read_text())["protocol_sha256"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    args = ap.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    idx = args.index
    if not 0 <= idx < len(protocol["eta_points"]):
        raise ValueError("η index out of frozen protocol range")
    state, eta = protocol["state"], protocol["eta_points"][idx]
    rt = make_runtime("ring_exchange")
    if rt.steps != 10:
        raise RuntimeError(f"unexpected pseudo horizon: {rt.steps}")
    outdir = HERE / "raw" / f"eta_{idx:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in protocol["seeds"]:
        for chain in CHAINS:
            path = outdir / f"seed_{seed:02d}_{chain}.json"
            if path.exists():
                continue
            result = run_one(rt, state, idx, np.asarray(eta, dtype=float), seed, chain)
            dump(path, result)
            print(json.dumps({k: result[k] for k in ("eta_index", "seed", "chain", "success", "termination", "elapsed_seconds")}), flush=True)
    print(json.dumps({"event": "eta_complete", "eta_index": idx}), flush=True)


if __name__ == "__main__":
    main()
