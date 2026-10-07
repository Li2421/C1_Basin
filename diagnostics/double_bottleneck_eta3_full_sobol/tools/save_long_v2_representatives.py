#!/usr/bin/env python3
"""Save deterministic R1-R4 eta=0/eta_rep trajectories for long v2."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"]="cpu"; os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np

from double_bottleneck.flowbc_4a_agent import load_checkpoint
from double_bottleneck.flowbc_4a_dataset import FlowBC4ADataset
from double_bottleneck.visualization import save_trajectory_svg
from diagnostics.double_bottleneck_eta3_full_sobol.tools.save_representatives import trace, summarize


ROOT=Path(__file__).resolve().parents[3]; STUDY=ROOT/"diagnostics/double_bottleneck_eta3_full_sobol"; LONG=STUDY/"long_run_v2"
CHECKPOINT=ROOT/"diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl"
DATASETS={"existing_untouched_test":ROOT/"diagnostics/double_bottleneck_initial_state_coverage/data/untouched_test_pool","fresh_untouched_test":ROOT/"diagnostics/double_bottleneck_sxl_baseline_maturation/data/fresh_test_pool"}


def main():
    selection=json.loads((LONG/"representative_selection.json").read_text()); basin={r["episode_id"]:r for r in json.loads((LONG/"timeout_basin_matrix.json").read_text())["episodes"]}; catalog={r["episode_id"]:r for r in json.loads((STUDY/"episode_catalog.json").read_text())["episodes"]}
    datasets={name:FlowBC4ADataset(path,"val",seed=45 if name.startswith("existing") else 46) for name,path in DATASETS.items()}; policy,_=load_checkpoint(CHECKPOINT,next(iter({d.environment_fingerprint for d in datasets.values()})))
    output=LONG/"representatives"; metadata=[]
    for role in ("R1","R2","R3","R4"):
        eid=selection[role]
        if eid is None: continue
        meta=catalog[eid]; dataset=datasets[meta["set"]]; episode=dataset.by_family[meta["family_id"]][0]; stem=f"{role}_{eid.replace('|','_')}"; traces={}
        pairs=[("eta0",(0.,0.,0.))]
        if basin[eid]["eta_rep"] is not None: pairs.append(("eta_rep",basin[eid]["eta_rep"]["theta"]))
        for label,theta in pairs:
            env,data=trace(policy,dataset,episode,theta,meta["seed"],meta["rollout_id"]); terminal=data.pop("terminal"); path=output/f"{stem}_{label}.npz"; np.savez_compressed(path,**data); svg=output/f"{stem}_{label}.svg"; save_trajectory_svg(env,data["positions"],theta,terminal,svg,title=f"{role} {eid} — {label}")
            traces[label]={"theta":list(map(float,theta)),"terminal":terminal,"npz":str(path.relative_to(ROOT)),"svg":str(svg.relative_to(ROOT)),"summary":summarize(data)}
        metadata.append({"role":role,"episode_id":eid,"rho":basin[eid]["rho"],"eta_rep":basin[eid]["eta_rep"],"traces":traces})
    result={"schema":"eta3_long_representatives_v1","episodes":metadata}; (output/"metadata.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"episodes":len(metadata),"traces":sum(len(r["traces"]) for r in metadata)},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
