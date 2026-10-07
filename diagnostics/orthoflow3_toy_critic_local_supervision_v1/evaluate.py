"""Run one frozen phase using the original scalar simulator and cache journals."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser(add_help=False);p.add_argument('--phase',choices=['val','test'],required=True)
known,_=p.parse_known_args();phase=known.phase
sys.argv.remove('--phase');sys.argv.remove(phase)
protocol=json.loads((OUT/'frozen_protocol.json').read_text())
source_path=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1/run_rollouts.py'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()==protocol['runner_sha256']
source=source_path.read_text()
old='"state_id": f"wide_ic_frozen_{ep:04d}", "episode_index": ep,'
new='"state_id": state["state_id"], "episode_index": ep, "parent_episode_index": state["parent_episode_index"], "h0": state["h_raw"], "physical_offset_m": state["physical_offset_m"],'
assert source.count(old)==1
source=source.replace(old,new)
runner=types.ModuleType('frozen_scalar_supervision_runner')
runner.__file__=str(source_path);sys.modules[runner.__name__]=runner
exec(compile(source,str(source_path),'exec'),runner.__dict__)
runner.OUT=OUT/phase
runner.EXPERIMENT_UID=protocol['experiment_uid']
def tasks():
    states=json.loads((OUT/phase/'frozen_proposals.json').read_text())['states']
    return [{'state':s,'kind':kind,'aliases':[kind],'eta':eta,'future_index':seed}
            for s in states for kind,eta in s['eta'].items() for seed in range(16)]
runner.tasks=tasks
if __name__=='__main__':runner.main()
