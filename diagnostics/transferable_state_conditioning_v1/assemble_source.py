"""Build fair physical-only/eta-only Q datasets from complete source labels.

The same rows feed both models; only the critic input differs. TEST is never
opened here. Each fold has its own source-only TRAIN and VALIDATION scenes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.physical_state_critic_formal_v1 import physical_entities

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent / "physical_state_critic_formal_v1"
SCENES = ("toy_give_way", "double_bottleneck", "four_way_intersection", "ring_exchange")
PANELS = {**{f"within_{s}": (s,) for s in SCENES},
          **{f"loso_{s}": tuple(x for x in SCENES if x != s) for s in SCENES}}
ETA_CENTER = np.array([.875, 0., .375], np.float32)
ETA_RADIUS = np.array([.375, .5, .375], np.float32)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(panel):
    assert panel in PANELS
    output = HERE / "source_data" / panel
    output.mkdir(parents=True, exist_ok=True)
    target = output / "trainval.npz"
    if target.exists():
        print(json.dumps({"panel": panel, "already_assembled": True, "sha256": sha(target)}))
        return
    rows = []
    source = []
    for split in ("train", "validation"):
        for scene in PANELS[panel]:
            path = FORMAL / "loso" / scene / split
            outcome = path / "outcomes.npz"
            d = np.load(outcome)
            assert d["seen"].all(), f"incomplete exact source labels: {path}"
            expected = (512, 16, 4, 5) if split == "train" else (32, 17, 16, 5)
            assert d["values"].shape == expected
            raw = d["values"][:, :16]
            success = np.sum((raw[..., 0] == 1) & (raw[..., 4] == 0), axis=2).astype(np.int16)
            numerical = np.sum(raw[..., 4] == 1, axis=2).astype(np.int16)
            failure = np.sum((raw[..., 0] == 0) & (raw[..., 4] == 0), axis=2).astype(np.int16)
            assert np.all(success + numerical + failure == expected[2])
            state_file = read(FORMAL / f"states_{scene}.json")["states"]
            by_alias = {s["alias"]: s for s in state_file if s["split"] == split}
            proposals = read(path / "proposals.json")["states"]
            assert len(proposals) == expected[0]
            assert [r["state_uid"] for r in proposals] == list(d["state_uids"])
            for i, p in enumerate(proposals):
                rows.append({"scene": scene, "split": split, "state_uid": p["state_uid"],
                             "physical": by_alias[p["state_alias"]]["physical"],
                             "eta": d["eta"][i, :16], "success": success[i],
                             "failure": failure[i], "numerical": numerical[i]})
            source.append({"scene": scene, "split": split,
                           "outcome_sha256": sha(outcome),
                           "proposal_sha256": sha(path / "proposals.json")})
    assert len({r["state_uid"] for r in rows}) == len(rows)
    entities = physical_entities.batch([{"scene": r["scene"], "physical": r["physical"]} for r in rows])
    assert np.all(entities["agents"][..., 3:7] == 0.)
    assert np.all(entities["agents"][..., 10:15] == 0.)
    assert np.all(entities["globals"] == 0.)
    eta = np.stack([r["eta"] for r in rows]).astype(np.float32)
    data = {"eta": (eta - ETA_CENTER) / ETA_RADIUS,
            "eta_raw": eta,
            "success": np.stack([r["success"] for r in rows]),
            "failure": np.stack([r["failure"] for r in rows]),
            "numerical": np.stack([r["numerical"] for r in rows]),
            "state_uids": np.asarray([r["state_uid"] for r in rows]),
            "scene": np.asarray([r["scene"] for r in rows]),
            "split": np.asarray([r["split"] for r in rows]),
            **{f"entity_{k}": v for k, v in entities.items()}}
    assert np.isfinite(data["eta"]).all()
    np.savez_compressed(target, **data)
    manifest = {"schema": "transferable_q_source_only_trainval_v1", "panel": panel,
                "source_scenes": list(PANELS[panel]), "train_states": int(np.sum(data["split"] == "train")),
                "validation_states": int(np.sum(data["split"] == "validation")),
                "trainval_sha256": sha(target), "source": source,
                "eta_center": ETA_CENTER.tolist(), "eta_radius": ETA_RADIUS.tolist(),
                "feature_definition": "masked physical entities; Flow/policy/controller/global channels zero",
                "feature_code_sha256": sha(physical_entities.__file__),
                "test_labels_opened": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"panel": panel, "train_states": manifest["train_states"],
                      "validation_states": manifest["validation_states"], "sha256": sha(target)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", choices=tuple(PANELS), required=True)
    assemble(p.parse_args().panel)
