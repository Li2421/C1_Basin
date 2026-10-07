#!/usr/bin/env python3
from audit import *
def main():
 hs=json.load(open(HERE/'hypothesis_status.json'));entry=dict(hypothesis_id='H7',family='bounded_quadratic_with_exclusion',mathematical_form='E_bridge AND g0(z)>=0 AND g1(z)>=0; quadratic A0<=-.001I; <=20coefficients',
  current_status='PROTOCOL_FROZEN_NOT_YET_FIT',key_support='20/21 fresh false inclusions outside prior positive hull;7/7 sampled extent transitions monotonic; one deep interior failure needs independent exclusion',key_failure=None,
  unresolved_property='Whether bounded support and one flexible exclusion preserve recall without fresh false inclusion',next_discriminating_test='Fit only after current acquisition checkpoint; same geometry gates; new retained validation only if cached screen passes')
 if not any(h['hypothesis_id']=='H7' for h in hs):hs.append(entry)
 dump('hypothesis_status.json',hs)
 state=json.load(open(HERE/'working_state.json'));state['actual_stage_jobs']['pocket_round3']=919
 state['next_action']='Finish902 DBgeometry2 and919 pocket scans. Aggregate exact evidence with approved numerical quarantine. Analyze intervals/corridor. Fit bounded_family.py only after complete inventory; frozen protocol in synthesized_families/bounded_quadratic/protocol.md. No accepted family, no training.'
 state['round2_properties']={'extent_segments':7,'monotonic_sampled_transitions':7,'new_internal_pocket_short_corridors':2,'all_failure_corridors':0,'internal_pocket_status':'UNRESOLVED_NOT_A_CONFIRMED_HOLE'}
 dump('working_state.json',state)
 history=json.load(open(HERE/'search_history.json'));existing={r['id'] for r in history['rounds']}
 for r in [dict(id='pocket_round3',job_id=919,new_Q64=18,status='RUNNING',purpose='Six axial directions and shortest known-negative graph detour for deep internal failure'),dict(id='synthesis_bounded_quadratic',status='PROTOCOL_FROZEN',protocol='synthesized_families/bounded_quadratic/protocol.md',parameters_max=20,why='Separate finite outer support from interior exclusion; no fitted result yet')]:
  if r['id'] not in existing:history['rounds'].append(r)
 dump('search_history.json',history)
 ei=json.load(open(HERE/'evidence_index.json'))
 for name in ['bounded_family.py','synthesized_families/bounded_quadratic/protocol.md','design_pocket_round3.py','success_failure_transition.csv','semialgebraic/fresh_failure_diagnosis.json','semialgebraic/fresh_validation_r1_gate.json']:
  ei['STAGE2:'+name]={'path':str(HERE/name),'sha256':sha(HERE/name)}
 dump('evidence_index.json',ei)
 print('Synthesis hypothesis and pocket test recorded; no scientific gates relaxed.')
if __name__=='__main__':main()
