#!/usr/bin/env python3
"""Freeze a cache-qualified independent replication cohort and screening plan."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1'); OUT=ROOT/'diagnostics/orthoflow3_mindef_independent_replication_v1'
M=ROOT/'diagnostics/orthoflow3_representation_migration_v1'; OLD=ROOT/'diagnostics/orthoflow3_analytic_mindef_stability_v1'; LOW=ROOT/'diagnostics/orthoflow3_low_frontier_enrichment_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if (OUT/'independent_active_state_manifest.json').exists():raise RuntimeError('frozen')
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 seq=list(csv.DictReader(open(LOW/'enrichment_eta_sequence.csv')))
 if len(seq)!=32 or [int(x['sobol_index']) for x in seq]!=list(range(256,288)):raise RuntimeError('invalid required sequence')
 # The migration subset is a pre-existing outcome-blind uniform subset of the 424 pool;
 # cache qualification means its zero/active B63 status is available without new discovery rollouts.
 selected=json.load(open(M/'migration_subset_manifest.json'))['selected_states'];summary={r['state_id']:r for r in csv.DictReader(open(M/'subset_basin_summary.csv'))}
 old_states=json.load(open(OLD/'state_manifest.json'))['states'];exclude=sorted({x['state_id'] for x in old_states})
 (OUT/'previous_state_exclusion_manifest.json').write_text(json.dumps({'excluded_state_ids':exclude,'source':'analytic_mindef_stability_v1 state manifest; low-frontier reused exactly this cohort','count':len(exclude)},indent=2)+'\n')
 eligible=[x for x in selected if x['state_id'] not in set(exclude)]
 rng=np.random.default_rng(2026092706); order=rng.permutation(len(eligible)).tolist(); ordered=[eligible[i] for i in order]
 permutation={'schema':'mindef_independent_cache_qualified_permutation_v1','selection_seed':2026092706,'population':'27 non-excluded states from the pre-existing random 32-state OrthoFlow3 migration subset; each has cached exact B63 classification','legacy_424_source_manifest':str(M/'migration_subset_manifest.json'),'ordered_state_ids':[x['state_id'] for x in ordered],'full_424_subset_indices':[x['state_index'] for x in ordered],'rule':'scan this order, accepting cached ACTIVE_REQUIRED status only; no proxy/J/canonical information participates in permutation'}
 (OUT/'frozen_candidate_state_permutation.json').write_text(json.dumps(permutation,indent=2)+'\n')
 accepted=[];scan=[]
 for pos,s in enumerate(ordered):
  status=summary[s['state_id']]['orthoflow3_status'];scan.append({'scan_rank':pos,'state_id':s['state_id'],'cached_status':status,'accepted':status=='ACTIVE_REQUIRED' and len(accepted)<10})
  if status=='ACTIVE_REQUIRED' and len(accepted)<10:accepted.append(s)
 # There are only nine independent active states in cache-qualified coverage.
 if len(accepted)<8:raise RuntimeError(('underresolved cached cohort',len(accepted)))
 states=[]
 for s in accepted:
  ss=summary[s['state_id']]
  states.append({**s,'cached_active_B63':int(ss['active_B63_candidates_confirmed']),'cached_zero_B63':ss['zero_B63']=='True','cached_status':ss['orthoflow3_status']})
 (OUT/'independent_active_state_manifest.json').write_text(json.dumps({'target':10,'achieved':len(states),'limitation':'cache-qualified pre-existing random migration subset contains only 9 non-excluded ACTIVE_REQUIRED states; expanding to unknown 424 states would require a new basin-discovery search projected beyond this replication budget','scan':scan,'states':states},indent=2)+'\n')
 old_sources={x.get('source_trajectory','') for x in old_states}; overlap=[]
 for s in states:
  overlap.append({'state_id':s['state_id'],'source_trajectory':s['source_trajectory'],'source_group_overlap':s['source_trajectory'] in old_sources})
 (OUT/'source_group_overlap_audit.json').write_text(json.dumps({'rows':overlap,'overlap_count':sum(x['source_group_overlap'] for x in overlap),'note':'state identity overlap is zero by construction'},indent=2)+'\n')
 (OUT/'enrichment_eta_sequence_reference.json').write_text(json.dumps({'path':str(LOW/'enrichment_eta_sequence.csv'),'sha256':sha(LOW/'enrichment_eta_sequence.csv'),'domain':{'eta1':[.5,1.25],'eta2':[-.5,.5],'eta3':[0,.75]},'sobol_indices':[256,287]},indent=2)+'\n')
 # 16 shared candidates are entirely outside the old 256 design and so untested for these states.
 arms=[]
 for s in states:
  for q in seq[:16]:
   eta=[float(q['eta1']),float(q['eta2']),float(q['eta3'])];i=int(q['sobol_index'])
   arms.append({'arm_id':f"SCREEN__{s['state_id']}__I{i}",'basis_family':'orthoflow3','state_id':s['state_id'],'state_file':s['state_file'],'state_sha256':s['state_sha256'],'absolute_step':s['absolute_step'],'rng_namespace':s['rng_namespace'],'eta':eta,'sobol_index':i,'seeds':[int(x) for x in s['matched_flow_seeds'][:8]],'role':'independent_shared_sobol_screen','anchor_rank':s['selection_rank'],'offset_steps':0,'probe_id':'SCREEN'})
 plan={'schema':'orthoflow3_mindef_independent_screen_v1','basis_family':'orthoflow3','stage':'screen16','selection':'first 16 not-yet-evaluated coordinates from exact low-frontier enrichment sequence; all selected independently before new proxy/J outcomes','arms':arms};plan['content_sha256']=dig(plan)
 (OUT/'screen_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 proto='''# Independent OrthoFlow3 minimum-deformation replication\n\nThe candidate sequence is byte-referenced from the completed low-frontier audit. The cohort is disjoint in state identity from that audit and comes from the pre-existing outcome-blind random 32-state subset of the authoritative 424-state manifest. Cached exact OrthoFlow3 B63 status permits a bounded scan; nine independent ACTIVE-required states exist after exclusion, meeting the protocol minimum of eight. New screening uses shared Sobol indices 256–271, then B63 promotion uses only index order for any baseline candidate and R_raw/R_gram for the two enrichment candidates; true J is not consulted for promotion.\n\nResource policy: up to two shards by default; up to six only at late night after confirming idle server. No neural model is trained.\n'''
 (OUT/'protocol.md').write_text(proto)
 print(json.dumps({'cache_qualified_pool':len(eligible),'scanned':len(scan),'independent_active':len(states),'screen_continuations':len(arms)*8,'step_upper':sum((850-a['absolute_step'])*8 for a in arms)},indent=2))
if __name__=='__main__':main()
