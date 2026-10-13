"""Source-only broad-η support ablation reusing exact Ring probe labels.

The 16 added Ring TRAIN states are disjoint from the stage-v1 Ring states.
No held-out target label or new rollout is used. The same rows feed both
state and eta-only critics in every augmented panel.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from diagnostics.physical_state_critic_formal_v1 import physical_entities
from diagnostics.transferable_state_conditioning_v1.assemble_source import (
    HERE, FORMAL, PANELS, ETA_CENTER, ETA_RADIUS, read, sha,
)

SOURCE = HERE / "stage_v1"
TARGET = HERE / "stage_v2_ring_bank"
SCENE = "ring_exchange"


def extra():
    panel = HERE / "source_anchor_probe" / SCENE
    cases = read(panel / "cases.json")
    assert len(cases) == 16*16*4
    proposals = read(FORMAL / "loso" / SCENE / "train" / "proposals.json")["states"][:16]
    physical = {r["alias"]: r["physical"] for r in read(FORMAL / f"states_{SCENE}.json")["states"]}
    success = np.zeros((16,16), np.int16)
    numerical = np.zeros((16,16), np.int16)
    for c in cases:
        r = read(panel / "raw" / f"{c['rollout_uid']}.json")
        assert r["case"] == c
        i,j = c["state_index"],c["eta_index"]
        success[i,j] += int(r["success"])
        numerical[i,j] += int(r["numerical_failure"])
    failure = 4-success-numerical
    assert np.all(failure >= 0)
    eta = np.broadcast_to(np.asarray(read(HERE / "frozen_splits.json")["common_anchor_eta"],
                                     np.float32), (16,16,3)).copy()
    x = physical_entities.batch([{"scene":SCENE,"physical":physical[p["state_alias"]]}
                                 for p in proposals])
    return {"eta": (eta-ETA_CENTER)/ETA_RADIUS, "eta_raw":eta,
            "success":success,"failure":failure,"numerical":numerical,
            "state_uids":np.asarray([p["state_uid"] for p in proposals]),
            "state_indices":np.arange(16),
            "scene":np.asarray([SCENE]*16),
            "split":np.asarray(["train"]*16),
            **{f"entity_{k}":v for k,v in x.items()}}


def pad_like(values, reference):
    assert len(values.shape)==len(reference.shape)
    pad=[(0,0)]
    for current,wanted in zip(values.shape[1:],reference.shape[1:]):
        assert current<=wanted
        pad.append((0,wanted-current))
    return np.pad(values,pad)


def build():
    TARGET.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SOURCE/"source_index_splits.json",TARGET/"source_index_splits.json")
    extra_rows=extra()
    summary={"schema":"source_only_ring_broad_eta_support_ablation_v1",
             "source_probe_signal_sha256":sha(HERE/"source_anchor_probe"/"signal.json"),
             "source_probe_cases_sha256":sha(HERE/"source_anchor_probe"/SCENE/"cases.json"),
             "added_independent_ring_states":16,"added_state_eta_pairs":256,
             "added_exact_cached_rollouts":1024,"new_rollouts_generated":0,
             "panels":{}}
    for panel,sources in PANELS.items():
        src=SOURCE/"source_data"/panel
        dst=TARGET/"source_data"/panel
        dst.mkdir(parents=True,exist_ok=True)
        if SCENE not in sources:
            shutil.copy2(src/"trainval.npz",dst/"trainval.npz")
            shutil.copy2(src/"manifest.json",dst/"manifest.json")
            fit_link=TARGET/"fits"/panel
            fit_link.parent.mkdir(parents=True,exist_ok=True)
            if not fit_link.exists():fit_link.symlink_to(SOURCE/"fits"/panel,target_is_directory=True)
            summary["panels"][panel]={"augmented":False,
                                      "trainval_sha256":sha(dst/"trainval.npz")}
            continue
        d=np.load(src/"trainval.npz")
        assert not set(extra_rows["state_uids"].tolist()) & set(d["state_uids"].tolist())
        data={}
        for key in d.files:
            old=d[key];new=extra_rows[key]
            if key.startswith("entity_"):
                new=pad_like(new,old)
            assert old.shape[1:]==new.shape[1:],(panel,key,old.shape,new.shape)
            data[key]=np.concatenate((old,new),axis=0)
        target=dst/"trainval.npz"
        np.savez_compressed(target,**data)
        manifest=read(src/"manifest.json")
        manifest.update(schema="source_only_development_stage_ring_bank_augmented_v1",
                        train_states=manifest["train_states"]+16,
                        trainval_sha256=sha(target),
                        augmentation={"scene":SCENE,"independent_states":16,
                                      "eta_candidates":16,"seed_slots":1024,
                                      "probe_cases_sha256":summary["source_probe_cases_sha256"]})
        (dst/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        summary["panels"][panel]={"augmented":True,
                                   "trainval_sha256":sha(target),
                                   "train_states":manifest["train_states"]}
    (TARGET/"augmentation_manifest.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":build()
