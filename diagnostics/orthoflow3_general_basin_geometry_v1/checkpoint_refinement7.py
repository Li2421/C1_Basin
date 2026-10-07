#!/usr/bin/env python3
from audit import *
def main():
 s=json.load(open(HERE/'working_state.json'))
 paths=[r for r in read(HERE/'connectivity_paths.csv') if r['kind']=='success_detour']
 result=dict(routes=len(paths),complete_routes=sum(r['finite_path_all_success']=='True' for r in paths),B63=sum(int(r['B63']) for r in paths),Q64=sum(int(r['exact_Q64']) for r in paths),interpretation='Five proposed two-leg detours failed at least one sampled link; this rejects those routes, not all possible paths or true connectedness.')
 dump('targeted_probe_rounds/success_detours/decision.json',result)
 s.update(current_stage='SAME_CAPACITY_TARGETED_REFINEMENT_R7',evidence_summary=json.load(open(HERE/'inventory_summary.json')),next_action='Fit F3 with rejected r6 validation as disclosed development and new detour evidence; do not alter capacity/erosion. If cached gates pass, freeze r7 fresh validation. Check unchanged DB instance before any validation reuse.',saturation_status='ONE_NON_IMPROVING_REFINEMENT: r5->r6 false7/72 unchanged; recall improvement<5pp. Detours strengthen irregular-link evidence but establish no new topology class. Another informative fixed-capacity refinement is needed before Section20A can apply.',pending_connectivity_test=None)
 s['completed_stages']+=['F3r6 rejected7/72; DB24 exact-independent validation reused','success_detours30Q64:16B63,0/5 complete sampled routes']
 dump('working_state.json',s)
 hs=json.load(open(HERE/'hypothesis_status.json'))
 for h in hs:
  if h.get('family')=='minimal_polyhedral_support_quadratic_exclusion':h.update(current_status='R6_REJECTED_R7_PREDECLARED',key_failure={'r5':{'false':7,'n':72},'r6':{'false':7,'n':72}},next_discriminating_test='Same-capacity r7 with new exact failed-link and retained-boundary evidence; independent retained validation if cached gates pass')
 dump('hypothesis_status.json',hs)
 history=json.load(open(HERE/'search_history.json'))
 for r in history['rounds']:
  if r['id']=='minimal_polyhedral_refined_r6':r.update(status='REJECTED',false=7,n=72,db_validation_reused=24)
  if r['id']=='success_detours':r.update(status='COMPLETE_NO_FULL_ROUTE',result=result)
 history['rounds'].append(dict(id='refinement_r7_design',status='FROZEN_BEFORE_FIT_AND_NEW_OUTCOMES',hypothesis='Boundary evidence-limited F3 fit versus practical same-capacity precision/coverage plateau',protocol='synthesized_families/minimal_polyhedral/protocol.md',new_protocol_parameters=False,scientific_decision='Whether another targeted evidence refinement changes morphology or improves recall/false-inclusion by>=5pp; does not imply saturation merely because a fit fails.'))
 dump('search_history.json',history)
 ei=json.load(open(HERE/'evidence_index.json'))
 for f in ['targeted_probe_rounds/success_detours/decision.json','targeted_probe_rounds/minimal_validation_r6/gate.json','targeted_probe_rounds/minimal_validation_r6/failure_diagnosis.json','development_validation_batches.json']:
  ei['R7:'+f]={'path':str(HERE/f),'sha256':sha(HERE/f)}
 dump('evidence_index.json',ei)
if __name__=='__main__':main()
