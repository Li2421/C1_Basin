#!/usr/bin/env python3
"""Final hash/regression audit for unattended eta3 long run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[3]; STUDY=ROOT/"diagnostics/double_bottleneck_eta3_full_sobol"; LONG=STUDY/"long_run_v2"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def tree_sha(path):
    d=hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file() and "__pycache__" not in x.parts and x.suffix!=".pyc"): d.update(str(p.relative_to(path)).encode()+b"\0"); d.update(p.read_bytes())
    return d.hexdigest()


def main():
    prereg=json.loads((LONG/"PREREGISTRATION.json").read_text()); frozen=prereg["frozen_hashes"]
    paths={"checkpoint":ROOT/"diagnostics/double_bottleneck_recovery_density_final/model/ckpt_selected.pkl","dataset_manifest":ROOT/"diagnostics/double_bottleneck_recovery_density_final/data/manifest.json","macflow":ROOT/"double_bottleneck/flowbc_4a_agent.py","environment":ROOT/"double_bottleneck/environment.py","hard_projection":ROOT/"shared_control/hard_projection.py","canonical_p0":ROOT/"shared_control/diagnostic_corrector.py","eta_points":STUDY/"eta_points.json"}
    actual={name:sha(path) for name,path in paths.items()}; actual["toy_giveway_source_tree"]=tree_sha(ROOT/"toy_giveway"); matches={name:actual[name]==frozen[name] for name in frozen}
    if not all(matches.values()): raise RuntimeError(f"frozen mismatch {matches}")
    env=os.environ.copy(); env["C1_PYTHON"]="/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python"; result=subprocess.run(["bash","scripts/test.sh"],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); (LONG/"logs/regression.log").write_text(result.stdout)
    output={"schema":"eta3_long_regression_v1","returncode":result.returncode,"passed":result.returncode==0,"expected_tests":79,"frozen_hashes":actual,"all_hashes_match":all(matches.values()),"toy_unchanged":matches["toy_giveway_source_tree"]}
    (LONG/"regression_results.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n"); manifest=json.loads((LONG/"run_manifest.json").read_text()); manifest.update({"state":"complete","regression_passed":output["passed"],"hashes_match":True}); (LONG/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); print(json.dumps(output,sort_keys=True)); return result.returncode


if __name__=="__main__": raise SystemExit(main())
