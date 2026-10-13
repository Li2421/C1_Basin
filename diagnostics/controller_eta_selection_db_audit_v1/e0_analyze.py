"""Pre-registered, read-only cross-fitted state/eta selection audit."""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

HERE = Path(__file__).resolve().parent
NBOOT = 2000


def write(path, x):
    Path(path).write_text(json.dumps(x, indent=2, ensure_ascii=False) + "\n")


def split_score(y, a, b, ix=None):
    if ix is None:
        ix = np.arange(y.shape[0])
    qa = y[ix][:, :, a].mean(axis=2)
    qb = y[ix][:, :, b].mean(axis=2)
    state_choice = qa.argmax(axis=1)
    fixed_choice = qa.mean(axis=0).argmax()
    idx = np.arange(len(ix))
    adaptive = qb[idx, state_choice]
    fixed = qb[:, fixed_choice]
    rb = (y[ix][:, :, b].sum(axis=2) >= 7)
    return {
        "fixed": float(fixed.mean()), "state": float(adaptive.mean()),
        "gain": float((adaptive - fixed).mean()),
        "robust_fixed": float(rb[:, fixed_choice].mean()),
        "robust_state": float(rb[idx, state_choice].mean()),
        "robust_gain": float((rb[idx, state_choice].astype(int) - rb[:, fixed_choice].astype(int)).mean()),
        "fixed_index": int(fixed_choice),
        "state_choice_indices": state_choice.tolist(),
    }


def covering(y):
    # Q16/B15 is descriptive only, not a selector label or continuous-basin claim.
    good = y.sum(axis=2) >= 15
    feasible = good.any(axis=1)
    n = int(feasible.sum())
    if not n:
        return {"feasible_states": 0, "best_single_coverage": 0,
                "greedy_cover_size": 0, "exact_min_cover_size": 0}
    m = good[feasible].astype(float)
    remaining = np.ones(n, dtype=bool)
    chosen = []
    while remaining.any():
        j = int(np.argmax(m[remaining].sum(axis=0)))
        chosen.append(j)
        remaining &= ~m[:, j].astype(bool)
    exact = None
    if y.shape[1] <= 20:
        res = milp(np.ones(y.shape[1]), integrality=np.ones(y.shape[1]),
                   bounds=Bounds(0, 1),
                   constraints=LinearConstraint(m, np.ones(n), np.full(n, np.inf)),
                   options={"time_limit": 20.0, "mip_rel_gap": 0.0})
        if res.success:
            exact = int(round(res.fun))
    return {"feasible_states": n,
            "best_single_coverage": float(m.sum(axis=0).max() / len(y)),
            "greedy_cover_size": len(chosen), "greedy_eta_indices": chosen,
            "exact_min_cover_size": exact,
            "exact_solver_status": "optimal" if exact is not None else "not_solved_or_E_gt20"}


def audit_panel(rec):
    f = np.load(rec["file"], allow_pickle=False)
    y = f["Y"].astype(np.uint8)
    etas = f["eta_uids"].astype(str).tolist()
    assert etas == sorted(etas), "Canonical lexical tie rule requires sorted eta UIDs"
    assert y.shape == (rec["n_states"], rec["n_eta"], 16)
    halves = [(list(range(0, 16, 2)), list(range(1, 16, 2))),
              (list(range(1, 16, 2)), list(range(0, 16, 2)))]
    base = [split_score(y, a, b) for a, b in halves]
    seed = int.from_bytes(hashlib.sha256(rec["panel_uid"].encode()).digest()[:8], "big") ^ 20261011
    rng = np.random.default_rng(seed)
    boot = np.empty((NBOOT, 2), float)
    n = len(y)
    for bi in range(NBOOT):
        ix = rng.integers(0, n, n)
        for k in range(2):
            vals = [split_score(y, a, b, ix) for a, b in halves]
            boot[bi, k] = np.mean([v["gain" if k == 0 else "robust_gain"] for v in vals])
    ci = np.quantile(boot, [.025, .975], axis=0)
    random_splits = []
    for k in range(4):
        perm = rng.permutation(16)
        scores = [split_score(y, perm[:8].tolist(), perm[8:].tolist()),
                  split_score(y, perm[8:].tolist(), perm[:8].tolist())]
        random_splits.append({"partition": k, "first_half_seed_indices": perm[:8].tolist(),
                              "mean_gain": float(np.mean([v["gain"] for v in scores])),
                              "robust_gain": float(np.mean([v["robust_gain"] for v in scores]))})
    full = y.mean(axis=2)
    mx = full.max(axis=1)
    ties = (full == mx[:, None]).sum(axis=1)
    all_fail = (y.sum(axis=2) == 0).all(axis=1)
    cover = covering(y)
    avg = lambda key: float(np.mean([v[key] for v in base]))
    fixed = avg("fixed")
    gain = avg("gain")
    rgain = avg("robust_gain")
    residual = (1 - fixed) - float(all_fail.mean())
    min_cover = cover["exact_min_cover_size"] or cover["greedy_cover_size"]
    qualifies = (n >= 12 and residual >= .05 and min_cover >= 2
                 and ((gain >= .05 and ci[0, 0] > 0)
                      or (rgain >= .05 and ci[0, 1] > 0)))
    return {"panel_uid": rec["panel_uid"], "scene": rec["scene"],
            "controller_uid": rec["controller_uid"], "n_states": n,
            "n_eta": len(etas), "n_seed_per_cell": 16,
            "eta_uids": etas, "state_uids": f["state_uids"].astype(str).tolist(),
            "selected_rollouts": rec["selected_rollouts"],
            "paired_even_odd": base,
            "heldout_fixed_success": fixed, "heldout_state_specific_success": avg("state"),
            "heldout_gain": gain, "heldout_gain_ci95": ci[:, 0].tolist(),
            "robust_fixed": avg("robust_fixed"), "robust_state_specific": avg("robust_state"),
            "robust_gain": rgain, "robust_gain_ci95": ci[:, 1].tolist(),
            "fixed_failure_U": 1 - fixed, "all_eta_fail_I": float(all_fail.mean()),
            "U_minus_I": residual,
            "full_label_optimistic_oracle_success": float(mx.mean()),
            "unique_full_label_best_states": int((ties == 1).sum()),
            "tied_full_label_best_states": int((ties > 1).sum()),
            "number_different_full_label_argmax_eta": int(len(set(full.argmax(axis=1).tolist()))),
            "even_odd_state_choice_agreement": float(np.mean(np.asarray(base[0]["state_choice_indices"]) == np.asarray(base[1]["state_choice_indices"]))),
            "B15_cover": cover, "random_split_sensitivity": random_splits,
            "preregistered_gate_pass": bool(qualifies),
            "interpretation": "cross-fitted state-specific reference, not deployable or continuous-eta oracle"}


def main():
    index = json.loads((HERE / "PANEL_INDEX.json").read_text())
    ready = [p for p in index if p["status"] == "ready"]
    out = []
    for i, rec in enumerate(ready, 1):
        p = audit_panel(rec)
        out.append(p)
        if i % 10 == 0:
            print(f"{i}/{len(ready)} analyzed; gates {sum(x['preregistered_gate_pass'] for x in out)}", flush=True)
    write(HERE / "E0_PANEL_RESULTS.json", out)
    print("Completed", len(out), "gate_pass", sum(x["preregistered_gate_pass"] for x in out), flush=True)


if __name__ == "__main__":
    main()
