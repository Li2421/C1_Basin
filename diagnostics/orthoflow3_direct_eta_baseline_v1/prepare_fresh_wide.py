#!/usr/bin/env python3
"""Freeze an outcome-blind fresh WIDE cohort after TEST evaluation is frozen."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
SOURCE=ROOT/'diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py'
REFERENCE=ROOT/'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json'
IC_SEED=2026092711; FLOW_SEED=2026092712; N=200

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def chash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,x): path.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def main():
 if not (HERE/'heldout_64seed_closedloop.csv').is_file(): raise RuntimeError('fresh WIDE may only freeze after TEST closed loop is frozen')
 if (HERE/'fresh_wide_manifest.json').exists(): raise RuntimeError('fresh manifest already frozen')
 import importlib.util
 spec=importlib.util.spec_from_file_location('fresh_source',SOURCE); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 reference=json.loads(REFERENCE.read_text()); fresh=mod.generate(IC_SEED,N)
 prior,identities,_,prior_ids,assets=mod.collect_prior()
 distances=np.linalg.norm(fresh[:,None].astype(float)-prior[None].astype(float),axis=(2,3))
 seeds={v for _,v,_ in identities}; ids={f'orthoflow3_direct_eta_fresh_v1_{i:04d}' for i in range(N)}
 collisions={'exact_initial':int(np.sum(distances==0)),'seed':sorted(seeds & {IC_SEED,FLOW_SEED}),'source_id':sorted(ids & prior_ids)}
 if collisions['exact_initial'] or collisions['seed'] or collisions['source_id']: raise RuntimeError(('fresh overlap',collisions))
 frozen=datetime.now(timezone.utc).isoformat(); overlap={'status':'PASS','fresh_episode_count':N,'collisions':collisions,'minimum_l2_distance_to_prior':float(distances.min()),'prior_initial_records_checked':len(prior),'assets_checked':len(set(assets)),'ic_seed':IC_SEED,'flow_seed':FLOW_SEED,'frozen_before_outcomes':True,'utc':frozen}; overlap['content_sha256']=chash(overlap);write(HERE/'fresh_wide_overlap_audit.json',overlap)
 manifest={'schema':'orthoflow3_direct_eta_fresh_wide_v1','frozen_before_rollout':True,'frozen_utc':frozen,'episode_count':N,'generator':{'implementation':'authoritative WIDE uniform generator; no rejection','ic_seed':IC_SEED,'x_absolute_uniform':[.55,1.05],'y_uniform':[-.025,.025],'source_script':str(SOURCE),'source_sha256':sha(SOURCE)},'flow_randomness':{'root_seed':FLOW_SEED,'semantics':f'episode_key=fold_in(PRNGKey({FLOW_SEED}), rollout_id); step_key=fold_in(episode_key, physical_step)','matched_across_controllers':True},'controller_protocol':{'controllers':['Safety','OrthoFlow3 Direct-eta'],'eta_prediction_steps':[0],'eta_fixed_for_episode':True,'basis_recomputed_each_step':True,'second_projection':True,'online_Q_or_J':False,'gate_or_cadence':False},'environment':reference['environment'],'cbf':reference['cbf'],'outcome_protocol':reference['outcome_protocol'],'checkpoint_sha256':json.loads((HERE/'selected_checkpoint.json').read_text())['checkpoint_sha256'],'overlap_audit_sha256':sha(HERE/'fresh_wide_overlap_audit.json'),'episodes':[{'episode_index':i,'rollout_id':i,'source_id':f'orthoflow3_direct_eta_fresh_v1_{i:04d}','initial_positions':p.tolist()} for i,p in enumerate(fresh)]};manifest['content_sha256']=chash(manifest);write(HERE/'fresh_wide_manifest.json',manifest)
 print(json.dumps({'status':'FROZEN','manifest_sha256':sha(HERE/'fresh_wide_manifest.json'),'overlap':overlap},indent=2))
if __name__=='__main__':main()
