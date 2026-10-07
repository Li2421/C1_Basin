"""Freeze sources for the minimal historical-only k=1 DAgger diagnostic."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=ROOT/'diagnostics/gphi_dagger_k1_diagnostic_v1'
DENSE=ROOT/'diagnostics/gphi_retrained_dense_strict_deadlock_v1'
CHECKPOINT=ROOT/'diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz'
EXPECTED='340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700'

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 HERE.mkdir(parents=True,exist_ok=True)
 for d in ('raw_k1','states_k1','tuples_k1','logs'): (HERE/d).mkdir(exist_ok=True)
 if sha(CHECKPOINT)!=EXPECTED: raise RuntimeError('checkpoint mismatch')
 src=json.load(open(DENSE/'source_manifest.json'))
 hist=[s for s in src['states'] if s['benchmark']=='historical']
 fresh=[s for s in src['states'] if s['benchmark']=='fresh_unseen']
 if len(hist)!=11 or len(fresh)!=6: raise RuntimeError('cohort mismatch')
 eta={r['state_id']:r['eta'] for r in src['etas_for_diagnostic_only']}
 out={
  'schema':'gphi_dagger_k1_source_v1','collector_checkpoint':str(CHECKPOINT),
  'collector_checkpoint_sha256':EXPECTED,'training_sources':hist,
  'fresh6_heldout_sources':fresh,'frozen_eta_by_state':eta,
  'robust_flow_seeds':list(range(95310001,95310065)),
  'source_manifest':str(DENSE/'source_manifest.json'),'source_manifest_sha256':sha(DENSE/'source_manifest.json'),
  'training_cohort':'historical 11 only','fresh6_access_before_checkpoint_freeze':False,
  'collection_depth_physical_steps':1,'eta_search':False,
 }
 (HERE/'source_manifest.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PREPARED','historical':11,'candidates':704,'checkpoint':EXPECTED},indent=2))
if __name__=='__main__': main()
