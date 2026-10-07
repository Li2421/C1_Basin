"""Freeze paired states and cache-aware first oracle stages before outcomes."""
import csv, hashlib, json, os
from pathlib import Path

ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=ROOT/'diagnostics/orthoflow3_bilateral_canonical_audit_v1'
CONT=ROOT/'diagnostics/orthoflow3_local_basin_continuity_v1'
MIG=ROOT/'diagnostics/orthoflow3_representation_migration_v1'
DESIGN=ROOT/'diagnostics/double_bottleneck_eta3_full_sobol/eta_points.json'
def sh(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dig(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def raw_cache():
 out={}
 for directory in (CONT/'raw').glob('*'):
  if not directory.is_dir(): continue
  for f in directory.glob('*.jsonl'):
   for line in f.read_text().splitlines():
    r=json.loads(line); key=(r['state_id'],tuple(r['eta']),int(r['seed']),int(r['rng_namespace']))
    if key in out:
     old=out[key]
     if (old['success'],old['outcome'],old['terminal_step']) != (r['success'],r['outcome'],r['terminal_step']): raise RuntimeError(('cache mismatch',key))
    else: out[key]=r
 return out
def main():
 if (HERE/'paired_state_manifest.json').exists(): raise RuntimeError('refusing to overwrite frozen audit')
 HERE.mkdir(parents=True,exist_ok=True); (HERE/'logs').mkdir(exist_ok=True)
 anchors=json.loads((CONT/'anchor_manifest.json').read_text())['anchors']
 ns=json.loads((CONT/'neighbor_manifest.json').read_text())['neighbors']
 pairs=[]
 for a in sorted(anchors,key=lambda x:int(x['anchor_rank'])):
  choices=[n for n in ns if int(n['anchor_rank'])==int(a['anchor_rank']) and int(n['offset_steps'])==4]
  if not choices: choices=[n for n in ns if int(n['anchor_rank'])==int(a['anchor_rank']) and int(n['offset_steps'])==-4]
  if len(choices)!=1: raise RuntimeError(('neighbor selection',a['anchor_rank'],len(choices)))
  b=choices[0]
  pairs.append({'pair_rank':int(a['anchor_rank']),'anchor_state_id':a['state_id'],'anchor_state_file':a['state_file'],'anchor_state_sha256':a['state_sha256'],'anchor_absolute_step':a['absolute_step'],'neighbor_state_id':b['neighbor_id'],'neighbor_state_file':b['state_file'],'neighbor_state_sha256':b['state_sha256'],'neighbor_absolute_step':b['absolute_step'],'offset_steps':b['offset_steps'],'matched_flow_seeds':b['matched_flow_seeds'],'rng_namespace':b['rng_namespace'],'selection_rule':'t+4 preferred; t-4 only if +4 unavailable; independent of all outcomes'})
 if len(pairs)!=12: raise RuntimeError(len(pairs))
 cache=raw_cache(); pts=json.loads(DESIGN.read_text())['points']
 def arm(prefix,pair,state,eta,eta_index,seeds,role):
  return {'arm_id':f'{prefix}__{pair["pair_rank"]:02d}__{eta_index if eta_index is not None else "ZERO"}','basis_family':'orthoflow3','state_id':state['id'],'state_file':state['file'],'state_sha256':state['sha'],'absolute_step':state['step'],'rng_namespace':pair['rng_namespace'],'eta':eta,'eta_index':eta_index,'seeds':seeds,'role':role,'anchor_rank':pair['pair_rank'],'offset_steps':pair['offset_steps'],'probe_id':prefix}
 global_arms=[]; zero_arms=[]; reused=[]
 for p in pairs:
  b={'id':p['neighbor_state_id'],'file':p['neighbor_state_file'],'sha':p['neighbor_state_sha256'],'step':p['neighbor_absolute_step']}
  first=int(p['matched_flow_seeds'][0])
  for i,point in enumerate(pts):
   eta=point['theta']; key=(b['id'],tuple(eta),first,int(p['rng_namespace']))
   if key in cache: reused.append({'kind':'global256_exact','pair_rank':p['pair_rank'],'state_id':b['id'],'eta_index':i,'seed':first,'source':'continuity_raw'})
   else: global_arms.append(arm('G1',p,b,eta,i,[first],'neighbor_global256'))
  zseeds=[]
  for seed in p['matched_flow_seeds'][:64]:
   key=(b['id'],(0.0,0.0,0.0),int(seed),int(p['rng_namespace']))
   if key in cache: reused.append({'kind':'zero64','pair_rank':p['pair_rank'],'state_id':b['id'],'eta_index':'ZERO','seed':int(seed),'source':'continuity_raw'})
   else:zseeds.append(int(seed))
  if zseeds: zero_arms.append(arm('Z64',p,b,[0.0,0.0,0.0],None,zseeds,'neighbor_zero64'))
 plan={'schema':'of3_bilateral_initial_v1','basis_family':'orthoflow3','stage':'neighbor_global256_and_zero64','arms':global_arms+zero_arms,'selection':'all 256 authoritative Sobol candidates at first matched seed plus zero to 64; cache tuple reuse exact state/eta/seed/rng only'};plan['content_sha256']=dig(plan)
 (HERE/'initial_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 (HERE/'paired_state_manifest.json').write_text(json.dumps({'schema':'of3_bilateral_pairs_v1','selection_before_outcomes':True,'pairs':pairs,'content_sha256':dig(pairs)},indent=2)+'\n')
 cfg=json.loads((MIG/'orthoflow3_search_config.json').read_text())
 auth={'orthoflow3_impl':str(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'),'orthoflow3_sha256':sh(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'),'basis_interface':str(ROOT/'shared_control/basis_families.py'),'basis_interface_sha256':sh(ROOT/'shared_control/basis_families.py'),'search_config':cfg,'source_scripts':{'candidate_screen':str(MIG/'prepare_candidate_screen.py'),'promotion1':str(MIG/'prepare_promotion_plan.py'),'promotion2':str(MIG/'prepare_secondary_promotion.py')}}
 (HERE/'authoritative_oracle_manifest.json').write_text(json.dumps(auth,indent=2,sort_keys=True)+'\n')
 (HERE/'cache_reuse.json').write_text(json.dumps({'initial_cached_tuples':len(reused),'initial_new_tuples':sum(len(a['seeds']) for a in plan['arms']),'reused':reused,'policy':'only exact state/eta/seed/rng continuation tuples'},indent=2)+'\n')
 (HERE/'protocol.md').write_text('# OrthoFlow3 bilateral canonical audit\n\nPairs are frozen before outcomes: the 12 prior anchors each use t+4, with t-4 fallback only if absent. The exact migration oracle is reproduced: 256-point first-seed screen; archived deterministic top-eight screen across 16 matched streams; B63 promotion; robust success first, mean successful J_def second, eta-index tie-break. Existing continuity continuations are reused only under exact state/eta/seed/RNG equality.\n\nThe audit is sequential only to resolve canonical candidates: additional promotion is limited to candidates whose 16-seed evidence leaves the minimum-J robust identity unresolved. No topology rebuild or learning is performed.\n')
 print(json.dumps({'pairs':len(pairs),'global_new':len(global_arms),'zero_new':sum(len(a['seeds']) for a in zero_arms),'cached':len(reused),'initial_steps_upper':sum((850-a['absolute_step'])*len(a['seeds']) for a in plan['arms'])},indent=2))
if __name__=='__main__':main()
