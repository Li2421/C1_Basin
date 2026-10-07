"""Source TRAIN only: common per-controller/eta Bernoulli null vs heterogeneity.

Parametric bootstrap refits the common rate each replicate. This tests existence
of state variation, not h predictability, and never certifies Q4 as B15.
"""
import numpy as np
from scipy.special import xlogy
from .data import OUT, CONTROLLERS, read, write, connect, canonical


def deviance(s, n):
    q = s / n
    p = s.sum(-1, keepdims=True) / n.sum(-1, keepdims=True)
    # xlogy(0,0)=0; entropy form handles p=0/1 without fake pseudocounts.
    return 2 * (xlogy(s, q) + xlogy(n-s, 1-q) - xlogy(s, p) - xlogy(n-s, 1-p)).sum(-1)


def main():
    d = dict(np.load(OUT / "dataset.npz"))
    pairs = read(OUT / "pairs.json")
    train = np.flatnonzero(d["split"] == "train")
    # Historical promotions have outcome-dependent n. Use only the four
    # predeclared standard trials, not the adaptive historical tail, for this
    # fixed-n binomial noise null. Drop incomplete numerical cells here only.
    first_s = np.zeros_like(d["success"], dtype=int)
    first_n = np.zeros_like(d["success"], dtype=int)
    keys = {canonical({"future_index": k}) for k in range(4)}
    with connect(True) as db:
        for c, profile in enumerate(read(OUT / "protocol.json")["profiles"]):
            for i in train:
                p = pairs[i]
                records = db.execute("SELECT seed_key,success,numerical_failure FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND conflict_quarantined=0 AND compatibility_quality='EXACT_REUSE'",
                                     (p["state_uid"], p["eta_uid"], profile["controller_uid"])).fetchall()
                good = [r for r in records if r["seed_key"] in keys and not r["numerical_failure"]]
                first_s[c, i] = sum(r["success"] for r in good)
                first_n[c, i] = len(good)
    eta_uids = sorted({pairs[i]["eta_uid"] for i in train})
    rows = []
    rng = np.random.default_rng(20261003)
    for c, controller in enumerate(CONTROLLERS):
        for eta_uid in eta_uids:
            ii = [i for i in train if pairs[i]["eta_uid"] == eta_uid]
            s = first_s[c, ii]
            n = first_n[c, ii]
            active = n == 4
            s, n = s[active], n[active]
            p = float(s.sum() / n.sum())
            observed = float(deviance(s, n))
            simulated = rng.binomial(n, p, size=(5000, len(n)))
            null = deviance(simulated, np.broadcast_to(n, simulated.shape))
            pvalue = float((1 + (null >= observed-1e-12).sum()) / 5001)
            q = s / n
            observed_variance = float(np.var(q, ddof=1))
            sampling_variance = float(np.mean(p*(1-p)/n))
            rows.append({"controller": controller, "eta_uid": eta_uid, "states": len(n),
                         "trials": int(n.sum()), "pooled_success_rate": p,
                         "observed_Q_variance": observed_variance,
                         "common_Q_null_sampling_variance": sampling_variance,
                         "excess_variance_estimate": observed_variance-sampling_variance,
                         "deviance": observed, "bootstrap_p": pvalue})
    order = np.argsort([r["bootstrap_p"] for r in rows])
    adjusted = np.minimum.accumulate((np.array([rows[i]["bootstrap_p"] for i in order])*len(rows)/np.arange(1, len(rows)+1))[::-1])[::-1]
    for i, value in zip(order, adjusted):
        rows[i]["BH_FDR_q"] = float(min(1., value))
        rows[i]["BY_arbitrary_dependence_q"] = float(min(1., value * np.sum(1./np.arange(1, len(rows)+1))))
    result = {"source_TRAIN_only": True, "target_labels_read": False, "new_continuations": 0,
              "null": "Within each fixed controller/exact eta, all TRAIN source families share one Bernoulli success rate",
              "bootstrap_replicates": 5000, "columns": len(rows),
              "trials_per_state": "first four predeclared standard seeds only; all four must be nonnumerical",
              "columns_rejecting_common_Q_at_BH_0_05": sum(r["BH_FDR_q"] <= .05 for r in rows),
              "columns_rejecting_common_Q_at_BY_0_05": sum(r["BY_arbitrary_dependence_q"] <= .05 for r in rows),
              "by_controller": {c: sum(r["controller"] == c and r["BH_FDR_q"] <= .05 for r in rows) for c in CONTROLLERS},
              "rows": rows,
              "limitations": ["A rejection demonstrates heterogeneity, not predictable h information or transferable structure.",
                              "No rejection can reflect low power with 24 states and mostly four trials per state.",
                              "Bernoulli independence is assumed within a pair; matched seeds create dependence across tested columns.",
                              "Numerical outcomes excluded, not converted to negatives; conclusions describe observed compatible trials."]}
    write(OUT / "source_count_noise_audit.json", result)
    print({k: v for k, v in result.items() if k != "rows"})


if __name__ == "__main__":
    main()
