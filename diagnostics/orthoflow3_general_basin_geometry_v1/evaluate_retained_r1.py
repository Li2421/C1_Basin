#!/usr/bin/env python3
from audit import *
from families import contains,polynomial
def main():
 name='retained_validation_r1';manifest=read(HERE/f'targeted_probe_rounds/{name}/manifest.csv')
 models={r['state_id']:r['model'] for r in json.load(open(HERE/'semialgebraic/frozen_validation_r1_parameters.json'))};raw=defaultdict(dict)
 for p in (HERE/'raw'/name).glob('shard*.jsonl'):
  for line in open(p):
   r=json.loads(line);assert not r.get('execution_error');assert r['h_conditioning_identifier']==STATES[r['state_id']]['h_conditioning_identifier']
   k=(r['state_id'],key(r['eta']));f=int(r['future_index'])
   if f in raw[k]:assert raw[k][f]['success']==r['success']
   raw[k][f]=r
 rows=[]
 for p in manifest:
  sid=p['state_id'];eta_v=json.loads(p['eta']);m=models[sid]
  assert hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()==p['model_sha256']
  assert contains(m,(np.array(eta_v)-AFF)/SCALE,True)[0]
  g=raw.get((sid,key(eta_v)),{});complete=set(g)==set(range(64));s=sum(r['success'] for r in g.values())
  rows.append(dict(state_id=sid,eta_key=key(eta_v),eta=json.dumps(eta_v),trials=len(g),successes=s,Q64=s/64 if complete else '',B63=s>=63 if complete else '',
   confirmed_false_inclusion=complete and s<63,deadlock=sum(r['outcome'] in ('safe_deadlock','deadlock','strict_deadlock') for r in g.values()),timeout=sum(r['outcome']=='timeout' for r in g.values()),model_sha256=p['model_sha256']))
 write('semialgebraic/fresh_retained_validation_r1.csv',rows)
 complete=all(r['trials']==64 for r in rows);false=sum(r['confirmed_false_inclusion'] for r in rows)
 result=dict(exact_Q64=sum(r['trials']==64 for r in rows),expected=len(rows),confirmed_false_inclusion=false,batch_complete=complete,
  validation_decision='REJECTED' if false else 'PASSED_THIS_BATCH_ONLY' if complete else 'PENDING',generalized_gate_pass=False,
  unvalidated_empty_instance='T0_WIDE_perm00_ep0082',second_scenario_retained_validation='NOT_RUN',no_post_validation_repair=True)
 dump('semialgebraic/fresh_validation_r1_gate.json',result);print(json.dumps(result))
if __name__=='__main__':main()
