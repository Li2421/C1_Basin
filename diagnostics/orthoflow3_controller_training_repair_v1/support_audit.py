"""Outcome-free support and input-path audit; target used descriptively only."""
import json
import numpy as np
from .data import OUT, read, write, probe


def main():
    states = read(OUT / "states.json")
    physical = read(OUT / "physical.json")
    target = read(probe.OUT / "held_controller_true_t0_v1/context_physical.json")
    train = [p for s, p in zip(states, physical) if s["split"] == "train"]
    def physical_stats(rows):
        result = {}
        for name, values in {
            "remaining_fraction": [r["physical"]["remaining_fraction"] for r in rows],
            "mean_goal_distance": [np.linalg.norm(np.asarray(r["physical"]["goals"])-r["physical"]["positions"], axis=1).mean() for r in rows],
            "mean_speed": [np.linalg.norm(r["physical"]["velocities"], axis=1).mean() for r in rows],
        }.items():
            result[name] = dict(zip(("min", "q25", "median", "q75", "max"), np.quantile(values, [0, .25, .5, .75, 1.]).tolist()))
        return result
    result = {"train_physical": physical_stats(train), "target_physical_metadata_only": physical_stats(target),
              "target_outcomes_opened_by_this_script": False,
              "target_statistics_enter_training_normalization": False}
    if (OUT / "dataset.npz").exists():
        d = dict(np.load(OUT / "dataset.npz"))
        tr = d["split"] == "train"
        cc = d["context"][:, tr][d["valid"][:, tr]]
        center, scale = cc.mean(0), np.maximum(cc.std(0), .05)
        contexts = sorted([r for i in range(8) for r in read(probe.OUT / f"held_controller_true_t0_v1/context_shard{i}of8.json")], key=lambda r:r["pair_index"])
        held = np.asarray([r["held"] for r in contexts])
        z = (held-center)/scale
        result["target_context_outside_source_support"] = {
            "any_abs_z_above_3_fraction": float((np.abs(z)>3).any(1).mean()),
            "max_abs_z_quantiles": np.quantile(np.abs(z).max(1), [0, .25, .5, .75, 1.]).tolist(),
            "nominal_closing_abs_z_median": float(np.median(np.abs(z[:, 4]))),
            "eta_closing_abs_z_median": float(np.median(np.abs(z[:, 14])))}
        result["controller_trial_mass"] = (d["success"][:, tr]+d["failure"][:, tr]).sum(1).tolist()
        result["controller_objective_mass"] = [1/3]*3
        result["context_feature_valid_fraction"] = np.isfinite(d["context"]).all(-1).mean(1).tolist()
    write(OUT / "support_audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
