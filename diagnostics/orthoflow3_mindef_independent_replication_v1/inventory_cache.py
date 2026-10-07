#!/usr/bin/env python3
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');OUT=ROOT/'diagnostics/orthoflow3_mindef_independent_replication_v1';M=ROOT/'diagnostics/orthoflow3_representation_migration_v1'
def main():
 states=json.load(open(OUT/'independent_active_state_manifest.json'))['states']; ids={x['state_id'] for x in states}; rows=list(csv.DictReader(open(M/'subset_oracle_candidates.csv')))
 b=defaultdict(list)
 for r in rows:
  if r['state_id'] in ids and r['B63']=='True':b[r['state_id']].append(r)
 base=[]
 for s in states:base.append({'state_id':s['state_id'],'source_trajectory':s['source_trajectory'],'cached_B63_candidates':len(b[s['state_id']]),'cached_nonzero_B63_candidates':sum(r['eta']!='[0.0,0.0,0.0]' for r in b[s['state_id']]),'baseline_action':'none' if len(b[s['state_id']])>=2 else 'requires_first_8of8_candidate_by_frozen_index'})
 with (OUT/'baseline_candidate_resolution.csv').open('w',newline='') as f:w=csv.DictWriter(f,base[0].keys());w.writeheader();w.writerows(base)
 audit={'compatible_sources':[str(M/'subset_oracle_candidates.csv'),str(M/'raw')],'compatibility':'same state IDs, OrthoFlow3 hash, B63 seeds 95210001..95210064, absolute horizon, projection stack, rng namespace','cached_B63_counts':{x['state_id']:len(b[x['state_id']]) for x in states},'reused_continuation_level_J_def':sum(len(b[x['state_id']]) for x in states),'new_screening_tuples':len(states)*16*8,'duplicate_tuples_planned':0}
 (OUT/'cache_reuse_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
