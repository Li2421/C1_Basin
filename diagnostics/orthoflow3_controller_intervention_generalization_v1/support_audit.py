"""Post-hoc physical/controller context support diagnosis, no model tuning."""
import json

import numpy as np

from .train import FOLDS, OUT, read, write


def run():
    d = dict(np.load(OUT / "training_dataset.npz"))
    rows = read(OUT / "training_rows.json")
    result = {}
    for fold, target_scene in FOLDS.items():
        norm = read(OUT / "models" / fold / "structured_full" / f"seed{read(OUT / 'models_frozen.json')['folds'][fold]['structured_full']['selected_seed']}" / "normalization.json")
        source_indices = [i for i, r in enumerate(rows) if r["scene"] != target_scene and r["split"] == "train"]
        unique = {}
        for i in source_indices:
            r = rows[i]
            unique.setdefault((r["state_uid"], r["controller_uid"]), i)
        source = d["context"][list(unique.values())]
        target = np.load(OUT / f"target_nominal_{fold}.npz")["context"]
        center = np.asarray(norm["context_center"])
        scale = np.asarray(norm["context_scale"])
        a = (source - center) / scale
        b = (target - center) / scale
        nearest = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(1)
        low, high = np.quantile(a, [.01, .99], axis=0)
        outside = (b < low) | (b > high)
        result[fold] = {"target_scene": target_scene, "source_controller_conditions": len(a),
                        "target_states": len(b),
                        "target_nearest_source_context_distance_median": float(np.median(nearest)),
                        "target_nearest_source_context_distance_p90": float(np.quantile(nearest, .9)),
                        "target_outside_source_1_99_feature_count_median": float(np.median(outside.sum(1))),
                        "target_states_any_feature_outside_1_99": int(np.any(outside, 1).sum()),
                        "target_states_ge_3_features_outside_1_99": int((outside.sum(1) >= 3).sum()),
                        "per_feature_outside_fraction": outside.mean(0).tolist(),
                        "target_nearest_source_distances": nearest.tolist()}
    write(OUT / "controller_context_support_audit.json", result)
    print(json.dumps({f: {k:v for k,v in r.items() if k not in ('target_nearest_source_distances','per_feature_outside_fraction')}
                      for f,r in result.items()}, indent=2))


if __name__ == "__main__":
    run()
