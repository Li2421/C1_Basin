#!/usr/bin/env python3
from audit import *
import subprocess
def main():
 w=json.load(open(HERE/'working_state.json'));e=json.load(open(HERE/'evidence_index.json'));hs=json.load(open(HERE/'hypothesis_status.json'))
 core=json.load(open(HERE/'common_core_summary.json'));assert all(r['tested']==40 for r in core)
 for h in hs:
  if h['hypothesis_id']=='H6':h.update(current_status='HIGH_COVERAGE_COMMON_PANEL_CONFIRMED_NOT_UNIVERSAL',key_support={'best_B63_states':35,'total_states':40},key_failure={'universal_candidates':0,'tested_candidates':10},
   unresolved_property='positive-volume common core and exclusion geometry',next_discriminating_test='targeted geometry_round1; finite panel is not volume certification')
 dump('hypothesis_status.json',hs)
 history=json.load(open(HERE/'search_history.json'))
 for r in history['rounds']:
  if r['id']=='common_core':r.update(status='COMPLETE',new_continuations=13232,physical_steps=6211596,best_panel_B63='35/40')
  if r['id']=='db_preflight':r.update(status='COMPLETE_VALID',new_continuations=8,new_Q64=0)
 history['operational_note']='Next independent batches use6GPU shards*1CPU +6CPU-only workers =12CPU, only while verified idle at night; no changes to physical/controller semantics.'
 dump('search_history.json',history)
 w['completed_stages'].append('common_core40x10 exact matrix');w['next_action']='Launch geometry_round1 and DB_geometry1 in parallel; total6GPU shards and12CPU maximum'
 w['anomalies'].append({'issue':'historical h hash aliases','resolution':'exact replay of batch1/32/64 reproduces all80728 historical hashes; max h difference2.595e-15 within original1e-10 tolerance; no conflicting duplicate outcomes'})
 dump('working_state.json',w)
 print('Recorded common-core completion, numerical conditioning diagnosis and next-stage resource policy')
if __name__=='__main__':main()
