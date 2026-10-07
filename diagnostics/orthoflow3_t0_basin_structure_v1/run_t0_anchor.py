#!/usr/bin/env python3
"""Run one true-t0 ball with the accepted continuous-ball oracle."""
from __future__ import annotations
import argparse,importlib.util,json,shutil,sys
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_t0_basin_structure_v1';SOURCE=ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1/continuous_pilot6.py';FUTURE_ROOT=2026092811
def load_patched():
 text=SOURCE.read_text()
 replacements={
  "cloud=sobol_bridge(eq,affine,scale,64)":"cloud=sobol_bridge(eq,affine,scale,96)",
  "entries=[(-1,z,'zero')]+[(i,cloud[i],'stage32') for i in range(32)]":"entries=[(-1,z,'zero')]+[(i,cloud[i],'stage48') for i in range(48)]",
  "common.extend(screen_candidates(sid,entries,'stage32'))":"common.extend(screen_candidates(sid,entries,'stage48'))",
  "stage_limit=32;stage_name='stage32'":"stage_limit=48;stage_name='stage48'",
  "if stage_limit==32:":"if stage_limit==48:",
  "[(i,cloud[i],'extension64') for i in range(32,64)]":"[(i,cloud[i],'extension96') for i in range(48,96)]",
  "'extension64'))":"'extension96'))",
  "stage_limit=64;stage_name='extension64'":"stage_limit=96;stage_name='extension96'",
  "for target in (.25,.60,.90):":"for target in (.25,.60,.90,.97):",
 }
 for old,new in replacements.items():
  if old not in text:raise RuntimeError(f'patch anchor missing: {old}')
  text=text.replace(old,new)
 spec=importlib.util.spec_from_loader('t0_continuous_runtime',loader=None);m=importlib.util.module_from_spec(spec);m.__file__=str(SOURCE);sys.modules[spec.name]=m;exec(compile(text,str(SOURCE),'exec'),m.__dict__);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--finalize-only',action='store_true');ap.add_argument('--cap-steps',type=int,default=1500000);a=ap.parse_args();states=json.load(open(HERE/'t0_state_manifest.json'))['attempted_states'];st=states[a.index];sid=st['state_id'];out=HERE/'anchor_runs'/sid;out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True)
 m=load_patched();m.HERE=out;m.STATE_IDS=[sid];m.QDIR=HERE/'synthetic_qdir';m.DDIR=HERE/'empty_prior';m.QGDIR=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1';m.CAP_CONT=5000;m.CAP_STEPS=a.cap_steps
 for n in ('ebridge_definition.json','ebridge_halfspaces.csv','eta_normalization.json','ebridge_sobol_sequence.csv','frozen_ray_directions.csv'):shutil.copy2(HERE/n,out/n)
 (out/'frozen_state_manifest.json').write_text(json.dumps({'states':[st]},indent=2,sort_keys=True)+'\n');(out/'cache_reuse_audit.json').write_text(json.dumps({'compatible_exact_tuples':0,'known_B63_state_eta':[],'known_nonB63_state_eta':[],'new_rollouts_launched_during_inventory':0},indent=2)+'\n');(out/'cost_preflight.json').write_text(json.dumps({'within_caps':True,'per_state_continuation_cap':5000,'per_state_step_cap':1500000},indent=2)+'\n')
 original=m.load_old_module
 def patched_old():
  old=original();old.FUTURE_ROOT=FUTURE_ROOT;old.CAP_CONT=5000;old.CAP_STEPS=a.cap_steps
  if a.finalize_only:
   def forbidden_rollout(*_args,**_kwargs):raise RuntimeError('finalize-only guard: missing rollout tuple')
   old.Oracle._run=forbidden_rollout
  return old
 m.load_old_module=patched_old
 m.run()
if __name__=='__main__':main()
