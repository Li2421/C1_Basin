"""Label-blind agent-permutation equivariance check of physical input."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnostics.physical_state_critic_formal_v1 import physical_entities

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / "physical_state_critic_formal_v1"
SCENES = ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange")


def main():
    rng = np.random.default_rng(20261008)
    result = {"schema": "physical_entity_agent_permutation_equivariance_v1", "scene": {}}
    for scene in SCENES:
        rows = json.loads((FORMAL / f"states_{scene}.json").read_text())["states"]
        max_abs = 0.
        checks = 0
        for row in rows[:32]:
            p = row["physical"]
            n = len(p["positions"])
            x = physical_entities.entities(scene, p)
            for _ in range(10):
                perm = rng.permutation(n)
                modified = dict(p)
                for key in ("positions", "velocities", "goals"):
                    modified[key] = np.asarray(p[key])[perm].tolist()
                y = physical_entities.entities(scene, modified)
                expected = {"agents": x["agents"][perm],
                            "pairs": x["pairs"][perm][:, perm],
                            "obstacles": x["obstacles"][perm],
                            "globals": x["globals"],
                            "agent_mask": x["agent_mask"],
                            "obstacle_mask": x["obstacle_mask"]}
                max_abs = max(max_abs, *(float(np.max(np.abs(y[k]-v))) for k, v in expected.items()))
                checks += 1
        result["scene"][scene] = {"states": 32, "permutations": checks,
                                   "max_abs_feature_error": max_abs,
                                   "pass": max_abs <= 1e-6}
        print(scene, result["scene"][scene])
    (HERE / "encoder_invariance_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    assert all(v["pass"] for v in result["scene"].values())


if __name__ == "__main__":
    main()
