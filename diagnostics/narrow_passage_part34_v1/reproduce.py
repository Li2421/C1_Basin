"""Read-only audit of archived Stage-2 Toy TT traces; CPU one-step probes only."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/home/zhihan/research/Basin_C1")
FROZEN = Path("/home/zhihan/research/Basin_C1_flow_field_poc_20261004")
BASE = FROZEN / "validation_stage2"
OUT = ROOT / "diagnostics/narrow_passage_part34_v1"
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(FROZEN / "source"))

import jax
import numpy as np
from validation_v2.adapters import OtherRuntime
from validation_v2.run_validation import key_for
from validation_stage2.factorial_controller import uid

jax.config.update("jax_enable_x64", True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def csvdump(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = json.loads((BASE / "protocol.json").read_text())
    cases = {x["id"]: x for x in protocol["cases"] if x["state"]["scenario"] == "toy_give_way"}
    manifest = {}
    for rel in ("source/single_integrator/environment.py", "source/single_integrator/cbf.py",
                "source/shared_control/basis_families.py", "validation_v2/adapters.py",
                "poc/field_controller.py", "validation_stage2/factorial_controller.py",
                "validation_stage2/protocol.json", "validation_v2/assets/toy_ckpt.pkl"):
        manifest[rel] = digest(FROZEN / rel)
    manifest["archive"] = str(BASE / "results")
    manifest["seed_rule"] = protocol["seed_rule"]
    manifest["source_protocol_sha256_matches"] = digest(FROZEN / "validation_v2/protocol.json") == protocol["parent_v2_protocol_sha256"]
    dump("provenance.json", manifest)

    # Agent 0 goes right, agent 1 goes left. Agent 1 is the bay yielding candidate.
    # s=p0x-p1x grows through passage; z=p1y-p0y grows when agent 1 enters top bay.
    D = 2 * .16 + .0001
    trajectories = []
    trace_index = {}
    for case_id, case in sorted(cases.items()):
        stem = BASE / "results" / (case_id + "_TT")
        if not stem.with_suffix(".json").exists() or not stem.with_suffix(".npz").exists():
            continue
        result = json.loads(stem.with_suffix(".json").read_text())
        with np.load(stem.with_suffix(".npz")) as array:
            pos = array["positions"].copy()
            actions = array["actions"].copy()
            keys = array["noise_keys"].copy()
            cert = array["certificates"].copy()
            goals = array["goals"].copy()
        assert result["state_uid"] == case["state"]["uid"]
        assert result["seed"] == case["seed"] and np.allclose(result["eta"], case["eta"])
        assert np.allclose(pos[0], case["state"]["physical"]["positions"])
        assert np.allclose(goals, case["state"]["physical"]["goals"])
        assert len(pos) == len(actions) + 1 == len(keys) + 1
        assert np.max(np.abs(np.diff(pos, axis=0) - .05 * actions)) < 1e-10
        s = pos[:, 0, 0] - pos[:, 1, 0]
        z = pos[:, 1, 1] - pos[:, 0, 1]
        yi = np.flatnonzero(z >= D)
        yi0 = np.flatnonzero(-z >= D)
        pi = np.flatnonzero(s >= D)
        y = int(yi[0]) if len(yi) else -1
        y0 = int(yi0[0]) if len(yi0) else -1
        p = int(pi[0]) if len(pi) else -1
        between = z[y:p+1] if y >= 0 and p >= y else np.array([])
        role = "agent1" if y >= 0 and (y0 < 0 or y <= y0) else ("agent0" if y0 >= 0 else "none")
        role_y = y if role == "agent1" else y0
        signed_z = z if role == "agent1" else -z
        role_between = signed_z[role_y:p+1] if role_y >= 0 and p >= role_y else np.array([])
        row = dict(id=case_id, state_uid=result["state_uid"], seed=result["seed"],
                   eta_name=result["eta_name"], eta=json.dumps(result["eta"]),
                   termination=result["termination"], success=result["success"], steps=result["steps"],
                   yield_step=y, pass_step=p, ordered_yield_pass=bool(y >= 0 and p > y),
                   yield_role=role, role_yield_step=role_y,
                   role_ordered_yield_pass=bool(role_y >= 0 and p > role_y),
                   min_role_clearance_yield_to_pass=float(role_between.min()) if len(role_between) else "",
                   max_z=float(z.max()), max_s=float(s.max()),
                   min_z_yield_to_pass=float(between.min()) if len(between) else "",
                   min_pair_center_distance=float(np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1).min()),
                   min_linear_residual=float(cert[:, 0].min()) if len(cert) else "",
                   max_speed_excess=float(cert[:, 1].max()) if len(cert) else "",
                   collision=bool(result["collision"]),
                   ever_deadlock_candidate=bool(result["plant_summary"].get("ever_candidate_deadlock")),
                   deadlock=bool(result["plant_summary"].get("deadlock")),
                   trace=str(stem.with_suffix(".npz")))
        trajectories.append(row)
        trace_index[case_id] = (case, result, pos, actions, keys)
    csvdump("trajectory_audit.csv", trajectories)

    rt = OtherRuntime("toy_give_way")
    # State choice uses only geometry from cached traces, with both archived outcomes
    # represented. No new physical step is executed by these probes.
    selected = []
    for term in ("success", "deadlock", "timeout"):
        pool = [r for r in trajectories if r["termination"] == term]
        for row in pool[:2]:
            case, result, pos, actions, keys = trace_index[row["id"]]
            s = pos[:-1, 0, 0] - pos[:-1, 1, 0]
            z = pos[:-1, 1, 1] - pos[:-1, 0, 1]
            metric = np.abs(s + .35) + np.abs(z - .18)
            t = int(np.argmin(metric))
            selected.append((row, t))
    # Exploratory branch check: first cached action in ID order with only pair CBF
    # active, both speed balls slack, and all wall rows strictly slack.
    from single_integrator.cbf import barrier_constraints, CBFConfig
    from single_integrator.environment import GiveWayEnv
    test_env = GiveWayEnv(rt.config)
    for row in trajectories:
        case, result, pos, actions, keys = trace_index[row["id"]]
        for t in range(0, len(actions), 3):
            test_env.positions = pos[t]
            A, lower, _ = barrier_constraints(test_env.snapshot(), CBFConfig())
            residual = A @ actions[t].ravel() - lower
            if (abs(residual[0]) < 1e-7 and np.min(residual[1:]) > 1e-5
                    and np.max(np.linalg.norm(actions[t], axis=1)) < .499):
                selected.append((row, t))
                break
        if len(selected) == 7:
            break
    for row in trajectories:
        if (row["termination"] != "success" or not row["ordered_yield_pass"]
                or json.loads(row["eta"])[1] >= -.05):
            continue
        case, result, pos, actions, keys = trace_index[row["id"]]
        for t in range(0, len(actions), 3):
            test_env.positions = pos[t]
            A, lower, _ = barrier_constraints(test_env.snapshot(), CBFConfig())
            residual = A @ actions[t].ravel() - lower
            if (abs(residual[0]) < 1e-7 and np.min(residual[1:]) > 1e-5
                    and np.max(np.linalg.norm(actions[t], axis=1)) < .499):
                selected.append((row, t))
                break
        if len(selected) == 10:
            break
    witness_row = next(r for r in trajectories if r["id"] == "2f142928c7c061e81d88")
    for t in (100, 150, 200, 230):
        selected.append((witness_row, t))
    manifest["checkpoint_matches_loaded_runtime"] = digest(rt.checkpoint) == manifest["validation_v2/assets/toy_ckpt.pkl"]
    manifest["stage2_TT_controller_uid"] = uid(rt, "TT")
    dump("provenance.json", manifest)
    probes = []
    for row, t in selected:
        case, result, pos, actions, keys = trace_index[row["id"]]
        env = rt.make_env(case["state"])
        env.positions = pos[t].copy()
        env.velocities = actions[t-1].copy() if t else np.zeros((2, 2))
        env.step_count = t
        eta = np.asarray(case["eta"], dtype=float)
        assert result["controller_uid"] == uid(rt, "TT")
        assert np.array_equal(np.asarray(key_for(case, t)), keys[t])
        key = jax.numpy.asarray(keys[t])
        try:
            _, safety, flow, safe, delta, _ = rt.prepare(env, key, eta)
            B = rt.basis.compute(env.positions, env.goals, safe, rt.config.max_speed).values
            eta0 = eta.copy(); eta0[1] = 0
            u0, status0 = safety.project(safe + sum(e*b for e,b in zip(eta0,B)))
            u1, status1 = safety.project(safe + delta)
            archived_action_error = float(np.max(np.abs(u1 - actions[t])))
            assert archived_action_error < 1e-10
            def ds_dz(u): return [float(u[0,0]-u[1,0]), float(u[1,1]-u[0,1])]
            residual = safety.A @ u1.ravel() - safety.lower
            active = ["pair" if i == 0 else f"agent{(i-1)//8}_wall{(i-1)%8}"
                      for i,x in enumerate(residual) if x <= 1e-7]
            active += [f"speed{i}" for i in range(2) if abs(np.linalg.norm(u1[i])-rt.config.max_speed) <= 1e-7]
            s = float(env.positions[0,0]-env.positions[1,0]); z = float(env.positions[1,1]-env.positions[0,1])
            raw = [ds_dz(b) for b in B]
            predicted_perp_derivative = raw[1][1] * s*s/(s*s+z*z)
            predicted_pair_exact = raw[1][1] - z*(s*raw[1][0]+z*raw[1][1])/(s*s+z*z)
            # Finite difference of the exact frozen projection, same safe anchor.
            h = 1e-4
            plus,_ = safety.project(safe + delta + h*B[1]); minus,_ = safety.project(safe + delta - h*B[1])
            rp,_ = safety.project(safe + delta + h*B[2]); rm,_ = safety.project(safe + delta - h*B[2])
            branch = active == ["pair"] and all(abs(x)>1e-5 for x in residual[1:])
            probes.append(dict(id=row["id"], state_uid=row["state_uid"], outcome=row["termination"],
                               timestep=t, key=keys[t].tolist(), eta=eta.tolist(), positions=env.positions.tolist(),
                               flow=flow.tolist(), a_safe=safe.tolist(), B=[b.tolist() for b in B],
                               raw_ds_dz=raw, u_perp0=u0.tolist(), u_full=u1.tolist(),
                               ds_dz_perp0=ds_dz(u0), ds_dz_full=ds_dz(u1),
                               perp_delta_dz=ds_dz(u1)[1]-ds_dz(u0)[1], active=active,
                               min_linear_residual=float(residual.min()),
                               speed_excess=float(max(np.linalg.norm(u1,axis=1))-rt.config.max_speed),
                               pair_only_regular=branch, predicted_perp_derivative=predicted_perp_derivative if branch else None,
                               predicted_pair_exact=predicted_pair_exact if branch else None,
                               measured_perp_derivative=(ds_dz(plus)[1]-ds_dz(minus)[1])/(2*h),
                               measured_rel_derivative=(ds_dz(rp)[1]-ds_dz(rm)[1])/(2*h),
                               projection_status=[status0["status"],status1["status"]]))
            probes[-1]["archived_action_max_error"] = archived_action_error
        except Exception as exc:
            probes.append(dict(id=row["id"], timestep=t, category="numerical_unknown", error=repr(exc)))
    dump("static_probes.json", probes)
    from single_integrator.environment import segment_distance, point_segment_distance
    row = witness_row
    case, result, pos, actions, keys = trace_index[row["id"]]
    env = rt.make_env(case["state"])
    wall_sweep = segment_distance(pos[:-1,:,None,:], pos[1:,:,None,:],
                                  env.walls[None,None,:,0,:], env.walls[None,None,:,1,:])
    wall_sweep -= rt.config.agent_radius + rt.config.wall_radius
    rel0, rel1 = pos[:-1,0]-pos[:-1,1], pos[1:,0]-pos[1:,1]
    pair_sweep = point_segment_distance(np.zeros(2), rel0, rel1) - 2*rt.config.agent_radius
    witness = dict(case=case, outcome=result, trajectory_audit=row,
                   selected_probes=[p for p in probes if p.get("id") == row["id"]],
                   swept_wall_surface_min=float(wall_sweep.min()),
                   swept_pair_surface_min=float(pair_sweep.min()),
                   max_archived_action_recompute_error=max(p.get("archived_action_max_error",0)
                                                       for p in probes if "error" not in p))
    dump("witness.json", witness)
    print(json.dumps(dict(archived=len(trajectories), expected=len(cases), static=len(probes),
                          numerical_unknown=sum("error" in p for p in probes)), indent=2))


if __name__ == "__main__": main()
