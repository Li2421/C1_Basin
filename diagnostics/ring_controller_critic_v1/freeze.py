"""Seal all model/code/prediction artifacts before external outcomes are opened."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert not (HERE / "MODEL_FREEZE.receipt.json").exists()
    assert (HERE / "fits/full_complete.json").exists()
    selections = json.loads((HERE / "fits/selected_steps.json").read_text())
    assert set(selections) == {"A0", "A1", "A2", "A3"}
    expected = 7 * 4 * 3 + 4 * 3 + 3 * 3 * 3
    assert len(list((HERE / "predictions").glob("*.npz"))) == expected, (len(list((HERE / "predictions").glob("*.npz"))), expected)
    assert (HERE / "predictions/G_E_FROZEN_CHOICES.json").exists()
    assert (HERE / "evaluate.py").exists()
    paths = [p for tree in (HERE / "fits", HERE / "predictions") for p in tree.rglob("*") if p.is_file()]
    paths += [HERE / f for f in ("fit.py", "evaluate.py", "freeze.py", "PREREGISTRATION.md", "PREREGISTRATION.receipt.json", "DATA_AUDIT.json")]
    receipt = {
        "status": "MODEL_RULES_HYPERPARAMETERS_FROZEN_BEFORE_EXTERNAL_OUTCOMES",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selected_steps": selections,
        "external_success_values_read": False,
        "external_unknown_values_read": False,
        "files": {str(p.relative_to(HERE)): sha(p) for p in sorted(set(paths))},
    }
    (HERE / "MODEL_FREEZE.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "files": len(paths), "steps": selections}))


if __name__ == "__main__":
    main()
