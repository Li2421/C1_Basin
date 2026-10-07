#!/usr/bin/env python3
from audit import *
def main():
 state=json.load(open(HERE/'working_state.json'));state['actual_stage_jobs'].update(polyhedral_validation_r4=949,db_polyhedral_validation_r4=950)
 state['evidence_summary']=json.load(open(HERE/'inventory_summary.json'));state['next_action']='Monitor949/950; audit then evaluate_retained.py polyhedral_validation_r4 and db_polyhedral_validation_r4. F1rejected22/72. Diagnose remaining shape property or saturation only after F2 fresh result; no arbitrary cycle cap and no training.'
 state['candidate_status']={'family':'polyhedral_support_quadratic_exclusion','gate':'CACHED_SCREEN_PASS_NOT_VALIDATED','adequate_states':12,'full_recall':.9466666666666667,'retained_recall':.4864864864864865,'parameters_max':22,'fresh_validation':'72new v4 points pending'}
 state['saturation_status']='Not declared. F1 refined Toy comparable metrics changed <5pp; F2 fresh result pending. Evaluate fixed-cohort comparisons, not changing cohort medians.'
 dump('working_state.json',state)
 hs=json.load(open(HERE/'hypothesis_status.json'))
 for h in hs:
  if h.get('hypothesis_id')=='H7':h.update(current_status='FROZEN_REFINED_INSTANCE_REJECTED22_OF72',key_failure={'toy_false':14,'toy_n':48,'db_false':8,'db_n':24,'all22_outside_prior_positive_hull':True},next_discriminating_test='F2 replaces curved support with sparse tilted support planes')
 hs.append(dict(hypothesis_id='H8',family='polyhedral_support_quadratic_exclusion',mathematical_form='E_bridge AND <=3 support planes AND one quadratic exclusion',current_status='CACHED_SCREEN_PASS_FRESH_PENDING',latest_gate_file='family_gates_polyhedral_r4.json',key_support=state['candidate_status'],key_failure=None,unresolved_property='Outer support extrapolation precision',next_discriminating_test='949/950 exactQ64 fresh retained points'))
 dump('hypothesis_status.json',hs)
 history=json.load(open(HERE/'search_history.json'))
 for r in history['rounds']:
  if r['id']=='bounded_refined_r3':r.update(status='REJECTED',fresh_false_inclusions=22,fresh_n=72)
 history['rounds'].append(dict(id='polyhedral_r4',status='CACHED_SCREEN_PASS_FRESH_PENDING',jobs=[949,950],new_Q64=72,protocol='synthesized_families/polyhedral_quadratic/protocol.md'))
 dump('search_history.json',history)
 ei=json.load(open(HERE/'evidence_index.json'))
 for name in ['polyhedral_family.py','synthesized_families/polyhedral_quadratic/protocol.md','development_validation_batches.json','family_gates_polyhedral_r4.json','polyhedral_unit_tests.json']:
  ei['F2:'+name]={'path':str(HERE/name),'sha256':sha(HERE/name)}
 dump('evidence_index.json',ei)
 dump('active_scenario_inventory_scope.json',{'active_environment_modules':['single_integrator/environment.py','double_bottleneck/environment.py'],'toy_giveway/environment.py':'identity-preserving compatibility exports, not an additional scenario','flowbc':'GiveWay policy/data package, not another environment','independent_compatible_scenarios':2,'archival_or_dyad_wrappers':'not counted as additional compatible scenarios'})
 print('Polyhedral candidate and72point prospective validation checkpointed.')
if __name__=='__main__':main()
