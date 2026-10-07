#!/usr/bin/env python3
from audit import *
def main():
 state=json.load(open(HERE/'working_state.json'));state['actual_stage_jobs'].update(bounded_validation_r2=925,db_independent_interpolation_r3=926)
 state['next_action']='Monitor925/926; aggregate safely; evaluate_retained.py bounded_validation_r2. If rejected, diagnose, do not repair/reuse validation. DBinterpolation new points predeclared heldout; complete12state evidence then fit DBparameters if candidate still alive.'
 state['evidence_summary']=json.load(open(HERE/'inventory_summary.json'))
 state['round3_properties']={'ep0195_native_eta1_labels':'FSFFSSS','eta2_labels':'FSSFFFF','eta3_labels':'FFFFFFF','failure_corridors_to_domain_boundary':3,'interpretation':'failure intrusion with sampled exterior paths, not confirmed enclosed hole','DB_Toy_common_eta_B63':'0/40'}
 state['candidate_status']={'family':'bounded_quadratic_with_exclusion','gate':'CACHED_SCREEN_PASS_ONLY','adequate_states':11,'full_recall':.9090909090909091,'retained_recall':.45714285714285713,'transfer_recall':.8648648648648649,'cached_false_inclusion':0,'fresh_validation':'48new Toy points pending','missing_general_gate':'12adequate states and cross-scenario fresh precision'}
 state['completed_stages']+=['pocket_round3:18new exactQ64; negative intrusion paths found','db_geometry2:88new exactQ64; shared Toymodes0/40DBB63','bounded synthesized family cached fit; all8Toy retained sampling nonempty']
 state['completed_stages']=list(dict.fromkeys(state['completed_stages']));dump('working_state.json',state)
 hs=json.load(open(HERE/'hypothesis_status.json'))
 for h in hs:
  if h.get('hypothesis_id')=='H7':h.update(current_status='CACHED_SCREEN_PASS_FRESH48_PENDING',latest_gate_file='family_gates_bounded_r2.json',key_support=state['candidate_status'],next_discriminating_test='925 retained validation;926 independent DBinterpolation')
 dump('hypothesis_status.json',hs)
 history=json.load(open(HERE/'search_history.json'))
 for r in history['rounds']:
  if r['id']=='pocket_round3':r.update(status='COMPLETE',result=state['round3_properties'])
  if r['id']=='synthesis_bounded_quadratic':r.update(status='CACHED_SCREEN_PASS_FRESH_PENDING',gate='family_gates_bounded_r2.json')
 history['rounds'].append(dict(id='bounded_validation_r2',job_id=925,new_Q64=48,status='RUNNING',parameters='targeted_probe_rounds/bounded_validation_r2/frozen_parameters.json'))
 history['rounds'].append(dict(id='db_independent_interpolation_r3',job_id=926,new_Q64=24,status='RUNNING',roles='all new points heldout before outcomes; old roles unchanged'))
 dump('search_history.json',history);print('Bounded-family pending stage checkpointed; no generalized acceptance.')
if __name__=='__main__':main()
