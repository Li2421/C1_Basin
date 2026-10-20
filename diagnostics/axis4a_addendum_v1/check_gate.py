from common import *

def main():
 checks=[]
 for c in read(HERE/'gate_CASES.json'):
  q=read(HERE/'gate/raw'/f"{c['rollout_uid']}.json");old=read(c['reference_path']);assert sha(c['reference_path'])==c['reference_sha256'];assert sha(old['trace_path'])==old['trace_sha256']
  with np.load(q['trace_path']) as z,np.load(old['trace_path']) as oz:
   a,b=z['positions'],oz['positions'];first=old.get('first_action')
   if first is None:first=oz['actions'][0] if len(oz['actions']) else None
   action=(q['first_action'] is None and first is None) or (q['first_action'] is not None and first is not None and np.asarray(q['first_action'],np.float64).tobytes()==np.asarray(first,np.float64).tobytes());v=dict(group=c['reference_group'],rollout_uid=c['rollout_uid'],reference_path=c['reference_path'],first_action_bytes=action,positions_bytes=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),terminal=q['outcome']==old['outcome'],steps=q['episode_length']==old['episode_length'],x64=q['jax_enable_x64']);v['passed']=all(v[k] for k in ['first_action_bytes','positions_bytes','terminal','steps']) and not v['x64'];checks.append(v)
 groups={g:dict(total=sum(x['group']==g for x in checks),passed=sum(x['group']==g and x['passed'] for x in checks)) for g in ['R10','R35_geometry']};passed=read(HERE/'CONFIG_UNIT_TEST.json')['passed'] and len(checks)==32 and all(x['passed'] for x in checks);write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),passed=passed,groups=groups,checks=checks,A_invalidated=not passed));seal('CONFIG_TRAJECTORY_UNIT_FROZEN.receipt.json',[HERE/'CONFIG_UNIT_TEST.json',HERE/'CONSISTENCY_GATE.json'],before_any_A_controls=True);print('Gate',passed,groups,flush=True)
if __name__=='__main__':main()
