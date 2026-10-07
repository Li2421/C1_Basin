#!/usr/bin/env python3
"""Compact stage memory, not an automatic scientific acceptance decision."""
from audit import *
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--jobs',required=True);a=ap.parse_args();jobs=json.loads(a.jobs)
 gate=json.load(open(HERE/f'family_gates_{a.tag}.json'));assert len(gate)==1;g=gate[0];family=g['family']
 state=json.load(open(HERE/'working_state.json'));state['actual_stage_jobs'].update(jobs);state['updated_local']=time.strftime('%Y-%m-%d %H:%M:%S %Z');state['evidence_summary']=json.load(open(HERE/'inventory_summary.json'))
 state['next_action']=f"Monitor{list(jobs.values())}; after completion audit.py, evaluate_retained.py for{list(jobs)}. Frozen fit{a.tag}. No accepted family; diagnose failures before another hypothesis/refit."
 state['candidate_status']=dict(family=family,fit_tag=a.tag,protocol=a.protocol,cached_gate=g,fresh_batches=list(jobs),acceptance='NOT_YET_VALIDATED')
 dump('working_state.json',state)
 hs=json.load(open(HERE/'hypothesis_status.json'))
 for h in hs:
  rejection=[]
  for p in (HERE/'targeted_probe_rounds').glob('*/gate.json'):
   gg=json.load(open(p))
   if gg.get('family')==h.get('family') and gg.get('status')=='REJECTED':rejection.append(dict(batch=p.parent.name,false=gg['false_inclusions'],n=gg['expected']))
  if rejection:h.update(current_status='PREVIOUS_FROZEN_VALIDATION_ATTEMPTS_REJECTED',key_failure=rejection)
 h=next((h for h in hs if h.get('family')==family),None)
 if h is None:h=dict(hypothesis_id='H'+str(max(int(x['hypothesis_id'][1:]) for x in hs)+1),family=family);hs.append(h)
 h.update(mathematical_form=a.protocol,current_status='CACHED_PASS_FRESH_VALIDATION_PENDING',latest_gate_file=f'family_gates_{a.tag}.json',key_support=g,unresolved_property='Prospective retained precision and full scientific gates',next_discriminating_test=list(jobs))
 dump('hypothesis_status.json',hs)
 history=json.load(open(HERE/'search_history.json'))
 if not any(r['id']==a.tag for r in history['rounds']):history['rounds'].append(dict(id=a.tag,protocol=a.protocol,status='CACHED_PASS_FRESH_PENDING',jobs=jobs,cached_gate=g))
 dump('search_history.json',history)
 ei=json.load(open(HERE/'evidence_index.json'))
 for name in [a.protocol,f'family_gates_{a.tag}.json',f'family_models_{a.tag}.json','extent_piece_diagnosis.json','minimal_polyhedral_offline_correction.json','minimal_family_unit_tests.json']:
  if (HERE/name).exists():ei['CURRENT:'+name]={'path':str(HERE/name),'sha256':sha(HERE/name)}
 dump('evidence_index.json',ei);print('Stage memory synced:',a.tag,'pending validation; gates unchanged')
if __name__=='__main__':main()
