"""Source-only validation and state mismatch for eta-prior residual Q."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.physical_state_critic_formal_v1.train_unified import select_key
from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, PANELS, read, sha
from diagnostics.transferable_state_conditioning_v1.analyze_validation import bootstrap
from diagnostics.transferable_state_conditioning_v1.train_baselines import SEEDS, NEAR_TIE
from diagnostics.transferable_state_conditioning_v1.train_residual import StateEtaResidual, prior_logits

HERE = Path(os.environ.get("TRANSFER_Q_WORK_ROOT", str(HERE)))

def evaluate(panel):
    assert panel in PANELS
    runs=[read(HERE/"fits"/panel/"state_residual"/f"seed{s}"/"complete.json") for s in SEEDS]
    best=min(runs,key=lambda r:(*select_key(r["selected"],NEAR_TIE),r["seed"]))
    d=np.load(HERE/"source_data"/panel/"trainval.npz")
    val=np.flatnonzero(d["split"]=="validation")
    scenes=np.asarray(d["scene"])[val]
    exposure=d["success"][val]+d["failure"][val]+d["numerical"][val]
    assert np.all(exposure==exposure.flat[0])
    nseeds=int(exposure.flat[0]);assert nseeds in (4,16)
    q=d["success"][val]/nseeds
    b15=d["success"][val]>=(15 if nseeds==16 else 4)
    x={key[7:]:jnp.asarray(d[key][val]) for key in d.files if key.startswith("entity_")}
    eta=jnp.asarray(d["eta"][val])
    prior,_=prior_logits(panel,d["eta"])
    base=jnp.asarray(prior[val])
    net=StateEtaResidual()
    params=net.init(jax.random.PRNGKey(best["seed"]),{k:v[:1] for k,v in x.items()},eta[:1])
    ck=HERE/"fits"/panel/"state_residual"/f"seed{best['seed']}"/"checkpoint.msgpack"
    assert sha(ck)==best["checkpoint_sha256"]
    params=serialization.from_bytes(params,ck.read_bytes())
    @jax.jit
    def apply(xx):
        return base+net.apply(params,xx,eta)
    matched=np.asarray(apply(x)).argmax(1)
    rng=np.random.default_rng(20261008)
    mismatch=np.empty((100,len(val)),np.int16)
    for rep in range(100):
        mapped=np.arange(len(val))
        for scene in sorted(set(scenes)):
            ix=np.flatnonzero(scenes==scene)
            order=rng.permutation(ix)
            mapped[order]=np.roll(order,rng.integers(1,len(ix)))
            assert np.all(mapped[ix]!=ix)
        mismatch[rep]=np.asarray(apply({k:v[mapped] for k,v in x.items()})).argmax(1)
    idx=np.arange(len(val))
    matched_q=q[idx,matched]
    mismatch_q=q[idx[None,:],mismatch]
    matched_b=b15[idx,matched]
    mismatch_b=b15[idx[None,:],mismatch]
    baseline=read(HERE/"fits"/panel/"source_validation_diagnostic.json")
    eta_choice=np.asarray(baseline["eta_choice"])
    concat_choice=np.asarray(baseline["matched_state_choice"])
    result={"schema":"source_only_residual_validation_mismatch_v1",
            "rollout_seeds_per_pair":nseeds,
            "robust_threshold_interpretation":"B15" if nseeds==16 else "4/4 provisional, not B15",
            "panel":panel,"validation_states":len(val),"residual_run":best,
            "eta_prior_checkpoint_sha256":best["eta_prior_checkpoint_sha256"],
            "matched_B15":int(matched_b.sum()),
            "mismatched_B15_mean":float(mismatch_b.sum(1).mean()),
            "matched_mean_success":float(matched_q.mean()),
            "mismatched_mean_success":float(mismatch_q.mean()),
            "eta_only_B15":int(b15[idx,eta_choice].sum()),
            "concat_B15":int(b15[idx,concat_choice].sum()),
            "paired":{"residual_minus_eta_B15":bootstrap(matched_b.astype(float)-b15[idx,eta_choice].astype(float)),
                      "residual_minus_concat_B15":bootstrap(matched_b.astype(float)-b15[idx,concat_choice].astype(float)),
                      "residual_minus_mismatch_B15":bootstrap(matched_b.astype(float)-mismatch_b.mean(0)),
                      "residual_minus_mismatch_mean_success":bootstrap(matched_q-mismatch_q.mean(0))},
            "matched_choice":matched.tolist(),"mismatch_choice":mismatch.tolist(),
            "source_data_sha256":sha(HERE/"source_data"/panel/"trainval.npz"),
            "test_labels_opened":False}
    target=HERE/"fits"/panel/"residual_source_validation_diagnostic.json"
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("panel","matched_B15","mismatched_B15_mean","eta_only_B15","concat_B15","paired")}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--panel",choices=tuple(PANELS),required=True)
    evaluate(p.parse_args().panel)
