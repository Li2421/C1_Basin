#!/usr/bin/env python3
"""Frozen DB OrthoFlow3 rollout runner, reusing the audited prior implementation."""
import importlib.util,json,sys
from pathlib import Path
H=Path(__file__).parent
SRC=H.parent/'orthoflow3_db_shared_mode_transfer_v1/run_db.py'
spec=importlib.util.spec_from_file_location('db_transfer_runner',SRC)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
mod.HERE=H
def load_states():
 return {s['state_id']:s for s in json.load(open(H/'hard_state_manifest.json'))['states']}
mod.load_states=load_states
if __name__=='__main__':mod.main()
