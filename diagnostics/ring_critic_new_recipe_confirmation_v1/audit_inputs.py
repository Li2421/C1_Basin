"""Read-only historical package and N1 dataset audit; writes only Round17 artifacts."""
import collections
import datetime
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
R14=ROOT/"diagnostics/ring_independent_flow_confirmation_v2"
R15=ROOT/"diagnostics/ring_controller_population_v1"
DATA=ROOT/"diagnostics/ring_exchange_stage1/base_u_v7_local_plus_drift_dataset"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")

def main():
    prereg=json.loads((HERE/"PREREGISTRATION.receipt.json").read_text())
    assert sha(HERE/"PREREGISTRATION.md")==prereg["sha256"]
    audit=json.loads((R15/"HISTORICAL_CODE_AUDIT.json").read_text())
    assert audit["passed"] and len(audit["files"])>=10
    mismatches=[]
    for item in audit["files"]:
        path=ROOT/item["file"]
        if not path.is_file() or sha(path)!=item["sha256"]:mismatches.append(item["file"])
    assert not mismatches,mismatches
    sys.path.insert(0,str(R14/"git_43b5264"))
    from new_benchmark_common.dataset import _digest,TRAJECTORY_SCHEMA
    manifest=DATA/"manifest.json"
    m=json.loads(manifest.read_text())
    assert m["complete"] and len(m["files"])==sum(m["counts"]["split"].values())==1792
    assert m["counts"]["split"]["train"]==1547 and m["counts"]["split"]["dev"]==185 and m["counts"]["split"]["test"]==60
    rows=[];fail=[]
    for row in m["files"]:
        path=DATA/row["file"]
        if not path.is_file():fail.append([row["file"],"MISSING"]);continue
        with np.load(path,allow_pickle=False) as z:
            states=z["states"];obs=z["observations"];act=z["actions"]
            meta=json.loads(str(z["metadata_json"].item()))
            valid=(str(z["schema"].item())==TRAJECTORY_SCHEMA and len(act)==row["length"]
                   and len(obs)==len(states)==len(act)+1 and _digest(states,obs,act)==row["trajectory_digest"]
                   and meta.get("trajectory_digest")==row["trajectory_digest"])
        if not valid:fail.append([row["file"],"CONTENT_MISMATCH"])
        rows.append(dict(file=row["file"],sha256=sha(path),bytes=path.stat().st_size))
    n1_available=not fail and len(rows)==1792
    write(HERE/"N1_DATASET_FILE_HASHES.json",dict(manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest),files=rows))
    old_plan=json.loads((R15/"TRAINING_PLAN.json").read_text())
    originals={x["run"]:x for x in old_plan if x["wave"]==1}
    assert len(originals)==14
    plan=[]
    for j in range(4):
        o=originals["base_u_v7_local_macflow"]
        plan.append(dict(group="N1",run=o["run"],name=f"base_u_v7_local_plus_drift__seed{j+1}",seed=o["original_seed"]+40001+j,
                         original_seed=o["original_seed"],dataset=str(DATA.relative_to(ROOT)),dataset_manifest_sha256=sha(manifest),
                         config=o["original_config"],original_best=o["original_best"],actor_hidden_dims=[256,256,256],steps=100000,
                         available=n1_available))
    for run,o in originals.items():
        if run=="base_u_plus_w_v4_wide_macflow":continue
        plan.append(dict(group="N2",run=run,name=f"{run}__arch512",seed=o["original_seed"]+40001,original_seed=o["original_seed"],
                         dataset=o["dataset"],dataset_manifest_sha256=o["dataset_manifest_sha256"],config=o["original_config"],
                         original_best=o["original_best"],actor_hidden_dims=[512,512,512,512],steps=o["steps"],available=True))
    assert len(plan)==17 and len({p["name"] for p in plan})==17
    for p in plan:
        assert sha(ROOT/p["dataset"]/"manifest.json")==p["dataset_manifest_sha256"]
        assert json.loads((ROOT/p["config"]).read_text())["steps"]==p["steps"]
        old=pickle.loads((ROOT/p["original_best"]).read_bytes())
        if p["group"]=="N1":assert tuple(old["config"]["actor_hidden_dims"])==(256,256,256)
    write(HERE/"TRAINING_PLAN.json",plan)
    evidence=dict(status="PASS" if n1_available else "N1_UNAVAILABLE",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        historical_code_audit=str((R15/"HISTORICAL_CODE_AUDIT.json").relative_to(ROOT)),historical_code_audit_sha256=sha(R15/"HISTORICAL_CODE_AUDIT.json"),
        historical_package_files_verified=len(audit["files"]),historical_package_byte_mismatches=mismatches,
        n1_manifest_sha256=sha(manifest),n1_manifest_complete=bool(m["complete"]),n1_file_count=len(m["files"]),n1_verified_files=len(rows),
        n1_failures=fail,n1_available=n1_available,training_plan_sha256=sha(HERE/"TRAINING_PLAN.json"),
        reusable=dict(round16_A0_A3_checkpoints="diagnostics/ring_controller_critic_v1/fits/full/",
                      round16_transforms="diagnostics/ring_controller_critic_v1/fits/full_transform.npz",
                      round16_GE="diagnostics/ring_controller_critic_v1/predictions/G_E_FROZEN_CHOICES.json",
                      round15_eight_arm_labels="diagnostics/ring_controller_population_v1/dataset.jsonl",
                      T1_T2_T3_labels="existing labeled menus, used only in A3-all source training"),
        gaps=["17 novel model checkpoints" if n1_available else "13 N2 novel model checkpoints; N1 dataset unavailable",
              "new-controller R5 probes","48 deduplicated unlabeled states","new-controller × state × arm × seed Q4 labels"],
        videos_exempt="pure Q experiment per AGENTS.md")
    write(HERE/"HISTORICAL_EVIDENCE_AUDIT.json",evidence)
    print(json.dumps(dict(status=evidence["status"],package_files=len(audit["files"]),n1_files=len(rows),plan=len(plan))))

if __name__=="__main__":main()
