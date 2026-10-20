from core import *
def consistency():
 cases=read(HERE/'gate_CASES.json');rows=[read(HERE/'gate/reference'/f"{c['rollout_uid']}.json") for c in cases];assert len(rows)==16 and all(r['full_trajectory_bitwise_equal'] for r in rows);write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),passed=True,cases=16,physical_reference_repeats=32,first_action_native_outcome_length_and_full_trace_bitwise=True,precision=False));print('TT gate PASS',flush=True)
def capability():
 cases=read(HERE/'pilot_CASES.json');names=list(dict.fromkeys(c['base_controller_name'] for c in cases));result=[];eligible=[]
 for n in names:
  lo=[];hi=[];unknown=[]
  for a in ['F10','F13']:
   rows=[read(HERE/'pilot/raw'/f"{c['rollout_uid']}.json") for c in cases if c['base_controller_name']==n and c['arm']==a];assert len(rows)==16;lo.append(sum(r['success'] for r in rows)/16);hi.append(sum(r['success']+r['numerical_failure'] for r in rows)/16);unknown.append(sum(r['numerical_failure'] for r in rows))
  low_excluded=max(hi)<.15;high_excluded=min(lo)>.98;uncertain=((max(lo)<.15<=max(hi)) or (min(lo)<=.98<min(hi)))
  status='EXCLUDED_LOW' if low_excluded else 'EXCLUDED_HIGH' if high_excluded else 'INCONCLUSIVE_UNKNOWN' if uncertain else 'ELIGIBLE'
  if status=='ELIGIBLE':eligible.append(n)
  result.append(dict(controller=n,arm_order=['F10','F13'],lower=lo,upper=hi,unknown=unknown,status=status))
 write(HERE/'CAPABILITY_GATE.json',dict(utc=utc(),controllers=result,eligible=eligible,per_class={k:sum(n.startswith(k) for n in eligible) for k in ['M','P']},no_replacement_or_tuning=True));print('CAPABILITY',eligible,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
