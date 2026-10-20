"""Train exactly one Round18 controller using the byte-audited historical package."""
import datetime
import hashlib
import inspect
import json
import os
import pickle
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
os.environ["TMPDIR"]=str(HERE/"tmp")
(HERE/"tmp").mkdir(exist_ok=True)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main(index):
    reg=json.loads((HERE/"PREREGISTRATION.receipt.json").read_text())
    assert sha(HERE/"PREREGISTRATION.md")==reg["sha256"]
    audit=json.loads((HERE/"HISTORICAL_EVIDENCE_AUDIT.json").read_text())
    assert audit["historical_training_package_files_verified"]==18
    item=json.loads((HERE/"TRAINING_PLAN.json").read_text())[index]
    r15=json.loads((ROOT/"diagnostics/ring_controller_population_v1/HISTORICAL_CODE_AUDIT.json").read_text())
    assert all(sha(ROOT/x["file"])==x["sha256"] for x in r15["files"])
    assert sha(ROOT/item["dataset"]/"manifest.json")==item["dataset_manifest_sha256"]
    sys.path.insert(0,str(ROOT/"diagnostics/ring_independent_flow_confirmation_v2/git_43b5264"))
    import jax
    jax.config.update("jax_enable_x64",False)
    from new_benchmark_common.training import train_stage1
    devices=[str(x) for x in jax.devices()]
    assert any(x.startswith("cuda") for x in devices) and not jax.config.x64_enabled,devices
    cfg=json.loads((ROOT/item["config"]).read_text())
    kwargs={k:cfg[k] for k in inspect.signature(train_stage1).parameters if k in cfg}
    kwargs.update(seed=item["seed"],actor_hidden_dims=tuple(item["actor_hidden_dims"]))
    assert kwargs["steps"]==item["steps"] and kwargs["batch_size"]==256
    original=pickle.loads((ROOT/item["original_best"]).read_bytes())
    assert tuple(item["actor_hidden_dims"]) in ((256,256,256),(384,384,384),(512,512,512,512))
    dest=HERE/"checkpoints"/item["name"]
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        done=dest/"ROUND18_TRAINING_RESULT.json"
        assert done.exists() and (dest/"best.pkl").exists(),f"Partial existing training directory: {dest}"
        print(json.dumps(dict(index=index,status="EXACT_LOCAL_REUSE",dest=str(dest))),flush=True);return
    receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),index=index,item=item,
                 prereg_sha256=reg["sha256"],training_plan_sha256=sha(HERE/"TRAINING_PLAN.json"),
                 package_audit_sha256=sha(ROOT/"diagnostics/ring_controller_population_v1/HISTORICAL_CODE_AUDIT.json"),
                 training_module=inspect.getfile(train_stage1),devices=devices,jax_enable_x64=False,kwargs=kwargs)
    (HERE/"training_receipts").mkdir(exist_ok=True)
    (HERE/"training_receipts"/f'{item["name"]}.json').write_text(json.dumps(receipt,indent=2)+"\n")
    result=train_stage1(ROOT/item["dataset"],dest,**kwargs)
    (dest/"ROUND18_TRAINING_RESULT.json").write_text(json.dumps(result,indent=2,default=str)+"\n")
    print(json.dumps(dict(index=index,name=item["name"],status="TRAINED",dest=str(dest),result=result),default=str),flush=True)

if __name__=="__main__":main(int(sys.argv[1]))
