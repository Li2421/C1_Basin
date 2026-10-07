#!/usr/bin/env python3
from audit import *
def main():
 state=json.load(open(HERE/'working_state.json'));state['actual_stage_jobs']['geometry_round2']=913
 state['next_action']='Monitor913 targeted extrapolation-boundary/pocket probes and902 DBcross-scenario geometry. First aggregate and diagnose, then synthesize bounded analytic support only if measured properties justify it. Current semialgebraic model REJECTED21/42; no accepted family.'
 state['fresh_failure_property']={'outside_prior_positive_hull':20,'failed_total':21,'inside_ellipsoid':15,'deep_internal_failure':'ep0195 Q64=.203125; no hole/notch conclusion yet'}
 state['saturation_status']='NOT_REACHED: prospective falsification materially changed inferred extent reliability; targeted property round2 running.'
 dump('working_state.json',state)
 history=json.load(open(HERE/'search_history.json'))
 dump('search_history_before_round2.json',history)
 # Preserve the historical schema; append explicit new events in a separate chronological ledger.
 dump('search_history_round2.json',[
  {'event':'round1_complete','evidence':'1722 exactQ64;147 DBexact,1 numerical-affected eta unresolved','geometry':'25/32 sampled star paths successful;24/24 Toy native lines single sampled interval;4/10 sampled negative paths reach boundary'},
  {'event':'all_negative_constrained_refit','gates':'family_gates_after_round1_complete.json','survivor':'one quadratic;10adequate states; cross-scenario12gate not yet met'},
  {'event':'prospective_retained_falsification','new_Q64':42,'false_inclusions':21,'family_status':'REJECTED','property':'20 failures outside prior positive hull;1deep interior; enclosing ellipsoid still includes15failures'},
  {'event':'targeted_property_round2','new_Q64':27,'decision':'extent boundary versus intrusion; no new analytic family synthesized before outcome','job':913},
  {'event':'DBcross_scenario_round2','new_Q64':88,'decision':'actual shared eta transfer and long conditional extents','job':902}
 ])
 for r in history['rounds']:
  if r['id']=='geometry_round1':r.update(status='COMPLETE',job_id=888,new_Q64=222,result='geometry_round1 summary in working_state.json')
 existing={r['id'] for r in history['rounds']}
 for event in json.load(open(HERE/'search_history_round2.json')):
  rid=event['event']
  if rid not in existing:history['rounds'].append(dict(id=rid,**event))
 history['stop_condition_satisfied']=False;history['latest_property_round']='targeted_property_round2';dump('search_history.json',history)
 print('Failure and next discriminating tests checkpointed.')
if __name__=='__main__':main()
