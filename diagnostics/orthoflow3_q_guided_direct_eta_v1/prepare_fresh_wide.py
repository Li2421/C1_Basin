#!/usr/bin/env python3
"""Freeze 400 outcome-blind WIDE episodes only after held-out TEST is frozen."""
import hashlib,importlib.util,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1';SOURCE=ROOT/'diagnostics/gphi_structured_eta_fresh_wide_v1/prepare_manifest.py';REFERENCE=ROOT/'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json';IC=2026092721;FLOW=2026092722;N=400
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ch(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 if not (HERE/'heldout_test_summary.json').is_file():raise RuntimeError('TEST must be frozen first')
 if (HERE/'fresh_wide_manifest.json').exists():raise RuntimeError('already frozen')
 spec=importlib.util.spec_from_file_location('src',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);fresh=m.generate(IC,N);prior,identities,_,prior_ids,assets=m.collect_prior();d=np.linalg.norm(fresh[:,None].astype(float)-prior[None].astype(float),axis=(2,3));ids={f'orthoflow3_qguided_fresh_v1_{i:04d}' for i in range(N)};coll={'exact_initial':int(np.sum(d==0)),'seed':sorted({v for _,v,_ in identities}&{IC,FLOW}),'source_id':sorted(ids&prior_ids)}
 if coll['exact_initial'] or coll['seed'] or coll['source_id']:raise RuntimeError(coll)
 now=datetime.now(timezone.utc).isoformat();audit={'status':'PASS','collisions':coll,'episodes':N,'prior_initials_checked':len(prior),'assets_checked':len(set(assets)),'minimum_prior_distance':float(d.min()),'frozen_before_outcomes':True};dump(HERE/'fresh_wide_overlap_audit.json',audit);ref=json.loads(REFERENCE.read_text());sel=json.loads((HERE/'selected_checkpoint.json').read_text());base=json.loads((ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1/selected_checkpoint.json').read_text());man={'schema':'orthoflow3_qguided_fresh_wide_v1','frozen_utc':now,'frozen_before_outcomes':True,'episode_count':N,'generator':{'ic_seed':IC,'implementation':'authoritative WIDE uniform generator, no rejection','source_sha256':sha(SOURCE)},'flow_randomness':{'root_seed':FLOW,'matched_across_controllers':True,'semantics':f'episode_key=fold_in(PRNGKey({FLOW}),rollout_id); step_key=fold_in(episode_key,physical_step)'},'environment':ref['environment'],'cbf':ref['cbf'],'controller_protocol':{'controllers':['Safety','MSE-only baseline','frozen-Q-guided'],'eta_once_at_step0':True,'eta_latched':True,'orthoflow3_recomputed_each_step':True,'second_projection':True,'Q_at_deployment':False,'J_or_search_or_gate':False},'baseline_checkpoint_sha256':base['checkpoint_sha256'],'guided_checkpoint_sha256':sel['checkpoint_sha256'],'overlap_audit_sha256':sha(HERE/'fresh_wide_overlap_audit.json'),'episodes':[{'episode_index':i,'rollout_id':i,'source_id':f'orthoflow3_qguided_fresh_v1_{i:04d}','initial_positions':p.tolist()} for i,p in enumerate(fresh)]};man['content_sha256']=ch(man);dump(HERE/'fresh_wide_manifest.json',man);print(json.dumps({'frozen':N,'sha256':sha(HERE/'fresh_wide_manifest.json'),'overlap':audit},indent=2))
if __name__=='__main__':main()
