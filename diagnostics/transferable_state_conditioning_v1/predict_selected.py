"""Predict both frozen proposal and common-bank final pools, label blind."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, PANELS, ETA_CENTER, ETA_RADIUS, read, sha
from diagnostics.transferable_state_conditioning_v1.predict_frozen import target_data, logits, derangement
from diagnostics.transferable_state_conditioning_v1.train_residual import StateEtaResidual


def residual_logits(info, x_numpy, eta_numpy):
    n=len(eta_numpy)
    x={k:jnp.asarray(v) for k,v in x_numpy.items()}
    eta=jnp.asarray((eta_numpy-ETA_CENTER)/ETA_RADIUS)
    net=StateEtaResidual()
    template=net.init(jax.random.PRNGKey(info["state_seed"]),
                      {k:v[:1] for k,v in x.items()},eta[:1])
    checkpoint=Path(info["state_checkpoint"])
    assert sha(checkpoint)==info["state_checkpoint_sha256"]
    params=serialization.from_bytes(template,checkpoint.read_bytes())
    @jax.jit
    def apply(xx):
        return net.apply(params,xx,eta)
    return x,apply


def predict(panel,bank):
    assert panel in PANELS and bank in ("proposal","anchor")
    freeze_file=HERE/"final_models_frozen.json"
    frozen=read(freeze_file)
    assert frozen["final_outcomes_opened"] is False
    info=frozen["panels"][panel]
    assert sha(HERE/"source_data"/panel/"trainval.npz")==info["source_data_sha256"]
    scene=panel.removeprefix("within_").removeprefix("loso_")
    ids,x,eta=target_data(scene,bank)
    eta_info={"seed":info["eta_seed"],"checkpoint":info["eta_checkpoint"],
              "checkpoint_sha256":info["eta_checkpoint_sha256"]}
    eta_logits=logits(eta_info,"eta_only_wide",x,eta)
    if info["state_kind"]=="state_concat":
        state_info={"seed":info["state_seed"],"checkpoint":info["state_checkpoint"],
                    "checkpoint_sha256":info["state_checkpoint_sha256"]}
        state_logits=logits(state_info,"state_concat",x,eta)
        get_mismatch=lambda mapped:logits(state_info,"state_concat",x,eta,mapped)
    else:
        assert info["state_kind"]=="state_residual"
        features,apply=residual_logits(info,x,eta)
        state_logits=eta_logits+np.asarray(apply(features))
        get_mismatch=lambda mapped:eta_logits+np.asarray(apply({k:v[mapped] for k,v in features.items()}))
    assert state_logits.shape==eta_logits.shape==(len(ids),16)
    assert np.isfinite(state_logits).all() and np.isfinite(eta_logits).all()
    rng=np.random.default_rng(20261008)
    permutations=np.stack([derangement(len(ids),rng) for _ in range(100)])
    mismatch=np.stack([get_mismatch(mapped).argmax(1) for mapped in permutations]).astype(np.int16)
    output=HERE/"predictions"/panel/f"{bank}.npz"
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,state_uids=np.asarray(ids),eta=eta,
                        state_logits=state_logits,eta_logits=eta_logits,
                        state_choice=state_logits.argmax(1),eta_choice=eta_logits.argmax(1),
                        mismatch_choice=mismatch,state_permutations=permutations)
    p=np.bincount(state_logits.argmax(1),minlength=16)/len(ids)
    audit={"panel":panel,"bank":bank,"states":len(ids),
           "chosen_state_model":info["state_kind"],
           "final_models_frozen_sha256":sha(freeze_file),
           "predictions_sha256":sha(output),
           "final_outcomes_opened":False,
           "state_selection_entropy":float(-(p[p>0]*np.log2(p[p>0])).sum())}
    (output.parent/f"{bank}_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--panel",choices=tuple(PANELS),required=True)
    p.add_argument("--bank",choices=("proposal","anchor"),required=True)
    a=p.parse_args()
    predict(a.panel,a.bank)
