#!/usr/bin/env python3
"""Check preregistration hashes, exact K, and nested uniform anchor sets."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


K_BY_VARIANT = {"U-Low": 64, "U-Mid": 256, "U-High": 640}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        episode = archive["source_episode_index"].astype(int)
        step = archive["source_step"].astype(int)
        rank = archive["selection_rank"].astype(int)
        attempt_sha = str(archive["cached_requery_sha256"])
    return set(zip(episode.tolist(), step.tolist(), strict=True)), episode, rank, attempt_sha


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    study = root / "diagnostics/double_bottleneck_uniform_recovery"
    prereg_sha = _sha(study / "PREREGISTRATION.json")
    manifest = json.loads((study / "data/manifest.json").read_text())
    training = json.loads((study / "models/training_comparison.json").read_text())
    evaluation = json.loads((study / "evaluation/comparison.json").read_text())
    checks = {
        "manifest_preregistration_hash": manifest["preregistration_sha256"] == prereg_sha,
        "evaluation_preregistration_hash": evaluation["preregistration_sha256"] == prereg_sha,
        "training_preregistration_hashes": all(
            row["preregistration_sha256"] == prereg_sha for row in training["variants"]
        ),
        "comparison_frozen_before_failure_inspection": evaluation[
            "individual_model_outcomes_inspected_before_freeze"
        ]
        is False,
    }
    split_details = {}
    for split, episodes in (("train", 72), ("val", 24)):
        variants = {}
        sets = {}
        for variant, count in K_BY_VARIANT.items():
            tag = variant.lower().replace("-", "_")
            path = study / "data" / f"{tag}_{split}.npz"
            pairs, episode, rank, cache_sha = _pairs(path)
            counts = Counter(episode.tolist())
            exact = len(counts) == episodes and set(counts.values()) == {count}
            unique = len(pairs) == episodes * count
            rank_ok = int(rank.min()) == 0 and int(rank.max()) == count - 1
            variants[variant] = {
                "rows": len(pairs),
                "exact_k_per_episode": exact,
                "unique_source_transitions": unique,
                "selection_rank_complete": rank_ok,
                "cache_sha256": cache_sha,
            }
            sets[variant] = pairs
        nested = sets["U-Low"] <= sets["U-Mid"] <= sets["U-High"]
        split_details[split] = {"variants": variants, "nested": nested}
        checks[f"{split}_exact_k"] = all(
            value["exact_k_per_episode"] for value in variants.values()
        )
        checks[f"{split}_unique"] = all(
            value["unique_source_transitions"] for value in variants.values()
        )
        checks[f"{split}_nested"] = nested
    checks["canonical_agent_hash_unchanged"] = (
        _sha(root / "double_bottleneck/flowbc_4a_agent.py")
        == "02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8"
    )
    result = {
        "schema": "double_bottleneck_uniform_recovery_protocol_validation_v1",
        "preregistration_sha256": prereg_sha,
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "splits": split_details,
    }
    if not result["all_checks_pass"]:
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    (study / "protocol_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
