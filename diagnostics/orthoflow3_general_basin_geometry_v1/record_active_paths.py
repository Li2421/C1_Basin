#!/usr/bin/env python3
from audit import *
def main():
 state=json.load(open(HERE/'working_state.json'));state['actual_stage_jobs']['success_detours']=979
 state['next_action']='Monitor973 new Toy validation and979 success detours (afterok973). DB24 independent validation reused only under validation_reuse_minimal_polyhedral_refined_r6.json. Aggregate then evaluate minimal_validation_r6 plus cached alias; analyze success_detours. No accepted general family yet.'
 state['validation_reuse_audit']='validation_reuse_minimal_polyhedral_refined_r6.json';state['cached_DB_validation_alias']='targeted_probe_rounds/db_minimal_validation_r6';state['pending_connectivity_test']='5 two-leg success detours,30new Q64; endpoints direct interpolations known non-B63'
 state['saturation_status']='NOT_REACHED: last comparable full12state validation improved16/72 to7/72 false inclusion (12.5pp). Same-family r6 pending.'
 dump('working_state.json',state)
 history=json.load(open(HERE/'search_history.json'))
 history['rounds'].append(dict(id='success_detours',job_id=979,dependency=973,new_Q64=30,purpose='Test sampled piecewise-linear connectivity around known failed straight links',status='FROZEN_AND_SUBMITTED'))
 for r in history['rounds']:
  if r['id']=='minimal_polyhedral_r5_corrected':r.update(status='REJECTED_GLOBALLY_DB_PASSED_UNCHANGED_INSTANCE',toy_false=7,toy_n=48,db_false=0,db_n=24)
 dump('search_history.json',history)
 ei=json.load(open(HERE/'evidence_index.json'))
 for name in ['validation_reuse_protocol.md','validation_reuse_minimal_polyhedral_refined_r6.json','targeted_probe_rounds/success_detours/selection.json','design_success_detours.py']:
  ei['LATEST:'+name]={'path':str(HERE/name),'sha256':sha(HERE/name)}
 dump('evidence_index.json',ei);print('Active validation reuse and piecewise connectivity test checkpointed.')
if __name__=='__main__':main()
