from common import *
def main():
 groups={}
 for c in read(HERE/'gate_CASES.json'):
  r=read(HERE/'gate/raw'/f"{c['rollout_uid']}.json");old=read(c['reference_path']);a=np.load(r['trace_path'])['positions'];b=np.load(old['trace_path'])['positions'];first=np.asarray(r['first_action'],np.float64).tobytes()==np.asarray(old['first_action'],np.float64).tobytes();checks=dict(rollout_uid=c['rollout_uid'],first_action_bytes=first,positions_bytes=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),outcome=r['outcome']==old['outcome'],steps=r['episode_length']==old['episode_length'],reference_path=c['reference_path']);checks['passed']=all(checks[k] for k in ['first_action_bytes','positions_bytes','outcome','steps']);groups.setdefault(c['cohort'],[]).append(checks)
 data={k:dict(passed=len(v)==16 and all(r['passed'] for r in v),checks=v,required=16) for k,v in groups.items()};write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),passed=all(v['passed'] for v in data.values()),groups=data));print({k:(v['passed'],sum(r['passed'] for r in v['checks'])) for k,v in data.items()})
if __name__=='__main__':main()
