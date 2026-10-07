"""Apply the category-balanced, source-grouped split without using labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from diagnostics.gphi_training_dataset_v1.build_states import assign_splits


HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = [json.loads(line) for line in (HERE / "state_manifest.jsonl").read_text().splitlines()]
    assignment = assign_splits(rows)
    for row in rows:
        row["split"] = assignment[row["leakage_group"]]
    (HERE / "state_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    split_manifest = {
        "assignment_unit": "leakage_group (source trajectory; all descendants of D1/D2/D4 anchor grouped)",
        "assignment_uses_oracle_labels": False,
        "assignment_rule": "fixed D1/D2/D4 root groups; category-only deterministic balancing",
        "state_ids": {
            split: [row["state_id"] for row in rows if row["split"] == split]
            for split in ("train", "validation", "test")
        },
        "leakage_groups": {
            split: sorted({row["leakage_group"] for row in rows if row["split"] == split})
            for split in ("train", "validation", "test")
        },
        "category_counts": {
            split: dict(Counter(row["category"] for row in rows if row["split"] == split))
            for split in ("train", "validation", "test")
        },
    }
    (HERE / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    print(json.dumps({
        "state_counts": {split: len(ids) for split, ids in split_manifest["state_ids"].items()},
        "category_counts": split_manifest["category_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
