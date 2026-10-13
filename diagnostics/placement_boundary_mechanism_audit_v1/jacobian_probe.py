"""Frozen TT/FF action Jacobians: independent directional probes and held-out Q.

The 48 state/Flow-key anchors are preselected in the immutable confirmation
manifest. This script does not run a physical rollout or alter a controller.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIELD = Path("/home/zhihan/research/Basin_C1_flow_field_poc_20261004")
HARD = Path("/home/zhihan/research/Basin_C1_hard_field_20261006")
CACHE = HARD / "transfer_mechanism_v2/state_dependence_control"
for path in (str(ROOT), str(FIELD), str(FIELD / "source"), str(FIELD / "source/scripts")):
    sys.path.insert(0, path)

import jax.numpy as jnp
from validation_stage2.run_rollouts import make_runtime


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def med(a):
    return float(np.median(a)) if len(a) else None


def principal_angles(a, b):
    ua, sa, _ = np.linalg.svd(a, full_matrices=False)
    ub, sb, _ = np.linalg.svd(b, full_matrices=False)
    ra = int(np.sum(sa > max(sa[0] * 1e-5, 1e-8))) if sa[0] else 0
    rb = int(np.sum(sb > max(sb[0] * 1e-5, 1e-8))) if sb[0] else 0
    if not ra or not rb:
        return [], ra, rb
    c = np.linalg.svd(ua[:, :ra].T @ ub[:, :rb], compute_uv=False)
    return np.degrees(np.arccos(np.clip(c, 0, 1))).tolist(), ra, rb


def action_from_frozen(rt, env, obs, key, safety, safe, delta, placement):
    if placement == "TT":
        return safety.project(safe + delta)[0]
    if placement == "FF":
        return rt.integrate(env, obs, key, delta, safety)[0]
    raise ValueError(placement)


def main():
    states = json.loads((HARD / "confirmation_manifest.json").read_text())["states"]
    state_by_uid = {s["uid"]: s for s in states}
    key_by_scene = json.loads((CACHE / "manifest.json").read_text())["actual_keys"]
    analysis, q_results, probe_rows = [], {}, []
    stable_q_results = {}
    raw_matrices = {}
    by_scene = {}
    for scene in ("toy_give_way", "ring_exchange"):
        uuids = sorted(s["uid"] for s in states if s["scenario"] == scene)
        rows = []
        for uid in uuids:
            z = np.load(CACHE / f"{uid}.npz")
            b = z["basis"]
            ms = z["center0_M_exec"][:, 2]
            gs = z["center0_J_eta_exec"][:, 2]
            tt, ff = gs[0], gs[3]
            angles, rank_tt, rank_ff = principal_angles(tt, ff)
            rows.append(dict(uid=uid, B=b, M_TT=ms[0], M_FF=ms[3], G_TT=tt, G_FF=ff))
            for name,value in [('B',b),('M_TT',ms[0]),('M_FF',ms[3]),('G_TT',tt),('G_FF',ff)]:
                raw_matrices[f'{uid}_{name}']=value
            analysis.append(dict(scene=scene, uid=uid, split="train" if uid in uuids[:12] else "test",
                sigma_TT=np.linalg.svd(tt, compute_uv=False).tolist(), sigma_FF=np.linalg.svd(ff, compute_uv=False).tolist(),
                rank_TT=rank_tt, rank_FF=rank_ff, principal_angles_deg=angles,
                relative_G_difference=float(np.linalg.norm(ff-tt)/max(np.linalg.norm(ff),1e-12)),
                chain_rule_TT=float(np.linalg.norm(tt-ms[0]@b)/max(np.linalg.norm(tt),1e-12)),
                chain_rule_FF=float(np.linalg.norm(ff-ms[3]@b)/max(np.linalg.norm(ff),1e-12))))
        by_scene[scene] = rows
        train, test = rows[:12], rows[12:]
        q = np.linalg.lstsq(np.concatenate([r["G_TT"] for r in train], axis=0),
                            np.concatenate([r["G_FF"] for r in train], axis=0), rcond=None)[0]
        def testset(group):
            residuals = [np.linalg.norm(r["G_FF"]-r["G_TT"]@q)/max(np.linalg.norm(r["G_FF"]),1e-12) for r in group]
            identity = [np.linalg.norm(r["G_FF"]-r["G_TT"])/max(np.linalg.norm(r["G_FF"]),1e-12) for r in group]
            qs = [np.linalg.lstsq(r["G_TT"], r["G_FF"], rcond=None)[0] for r in group]
            solo = [np.linalg.norm(r["G_FF"]-r["G_TT"]@v)/max(np.linalg.norm(r["G_FF"]),1e-12) for r,v in zip(group,qs)]
            return dict(global_relative_residual=residuals, identity_relative_residual=identity,
                        per_state_relative_residual=solo, individual_Q=[v.tolist() for v in qs],
                        median_global=med(residuals), median_identity=med(identity), median_per_state=med(solo))
        q_results[scene] = dict(train_uids=uuids[:12], test_uids=uuids[12:], Q=q.tolist(), train=testset(train), test=testset(test))
        quality={uid:json.loads((CACHE/f'{uid}.json').read_text())['diagnostics']['chains'] for uid in uuids}
        stable_train=[r for r in train if quality[r['uid']]['TT']['fd_stable'] and quality[r['uid']]['FF']['fd_stable']]
        stable_test=[r for r in test if quality[r['uid']]['TT']['fd_stable'] and quality[r['uid']]['FF']['fd_stable']]
        qs=np.linalg.lstsq(np.concatenate([r['G_TT'] for r in stable_train],axis=0),
                            np.concatenate([r['G_FF'] for r in stable_train],axis=0),rcond=None)[0]
        def score_stable(group):
            residual=[np.linalg.norm(r['G_FF']-r['G_TT']@qs)/max(np.linalg.norm(r['G_FF']),1e-12) for r in group]
            individual=[np.linalg.norm(r['G_FF']-r['G_TT']@np.linalg.lstsq(r['G_TT'],r['G_FF'],rcond=None)[0])/
                        max(np.linalg.norm(r['G_FF']),1e-12) for r in group]
            return dict(relative_residual=residual,per_state_best_residual=individual,
                        median_relative_residual=med(residual),median_per_state_best=med(individual))
        stable_q_results[scene]=dict(train_uids=[r['uid'] for r in stable_train],test_uids=[r['uid'] for r in stable_test],
            Q=qs.tolist(),train=score_stable(stable_train),test=score_stable(stable_test),
            stable_train_count=len(stable_train),stable_test_count=len(stable_test))

        # Directional vectors were fixed before probing, independent of FD axes.
        rng = np.random.default_rng(202610081 if scene == "toy_give_way" else 202610082)
        v = rng.normal(size=3);v /= np.linalg.norm(v)
        for uid in (uuids[0], uuids[12]):
            rt = make_runtime(scene);env = rt.make_env(state_by_uid[uid]);key = jnp.asarray(key_by_scene[scene], dtype=jnp.uint32)
            obs, safety, _, safe, _, _ = rt.prepare(env, key, np.zeros(3))
            row = next(r for r in rows if r["uid"] == uid)
            base = {p: action_from_frozen(rt, env, obs, key, safety, safe, np.zeros_like(safe), p).ravel() for p in ("TT","FF")}
            for p in ("TT", "FF"):
                g = row["G_"+p]; m = row["M_"+p]; b = row["B"]
                for eps in (1e-2, 3e-3, 1e-3, 3e-4):
                    plus = action_from_frozen(rt,env,obs,key,safety,safe,(b@(eps*v)).reshape(safe.shape),p).ravel()
                    minus = action_from_frozen(rt,env,obs,key,safety,safe,(b@(-eps*v)).reshape(safe.shape),p).ravel()
                    actual = plus-base[p];pred = eps*g@v
                    fd=(plus-minus)/(2*eps)
                    probe_rows.append(dict(scene=scene,uid=uid,placement=p,epsilon=eps,direction=v.tolist(),
                        absolute_error=float(np.linalg.norm(actual-pred)),
                        relative_error=float(np.linalg.norm(actual-pred)/max(np.linalg.norm(actual),1e-12)),
                        cosine=float(np.dot(actual,pred)/max(np.linalg.norm(actual)*np.linalg.norm(pred),1e-12)),
                        directional_fd_error=float(np.linalg.norm(fd-g@v)/max(np.linalg.norm(g@v),1e-12)),
                        M_B_chain_error=float(np.linalg.norm(g-m@b)/max(np.linalg.norm(g),1e-12))))
    np.savez_compressed(HERE/'raw_G_M.npz',**raw_matrices)
    save("raw_G_M.json", analysis)
    save("fixed_Q_train_test.json", q_results)
    save("stable_only_fixed_Q.json", stable_q_results)
    save("independent_directional_probes.json", probe_rows)
    summary = {}
    for scene in by_scene:
        aa = [r for r in analysis if r["scene"] == scene]
        pp = [r for r in probe_rows if r["scene"] == scene]
        summary[scene] = dict(states=len(aa),median_relative_G_difference=med([r["relative_G_difference"] for r in aa]),
            median_sigma_TT=np.median([r["sigma_TT"] for r in aa],axis=0).tolist(),
            median_sigma_FF=np.median([r["sigma_FF"] for r in aa],axis=0).tolist(),
            rank_pairs={f"{a}:{b}":sum(r["rank_TT"]==a and r["rank_FF"]==b for r in aa) for a in range(4) for b in range(4)},
            median_max_principal_angle_deg=med([max(r["principal_angles_deg"]) for r in aa if r["principal_angles_deg"]]),
            independent_probe={str(e):dict(n=sum(r["epsilon"]==e for r in pp),median_relative_error=med([r["relative_error"] for r in pp if r["epsilon"]==e]),
                median_directional_fd_error=med([r["directional_fd_error"] for r in pp if r["epsilon"]==e])) for e in (1e-2,3e-3,1e-3,3e-4)},
            heldout_Q=q_results[scene]["test"]["median_global"])
    save("gate_ABC_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
