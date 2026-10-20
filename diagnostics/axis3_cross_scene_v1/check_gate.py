from common import *
def main():
 result={sc:[] for sc in SCENES}
 for c in read(HERE/'gate_v2_CASES.json'):
  r=read(HERE/'gate_v2/raw'/f"{c['rollout_uid']}.json");old=read(c['reference_path']);a=np.load(r['trace_path'])['positions'];b=np.load(old['trace_path'])['positions'];first=(r['first_action'] is None and old['first_action'] is None) or (r['first_action'] is not None and old['first_action'] is not None and np.asarray(r['first_action'],np.float64).tobytes()==np.asarray(old['first_action'],np.float64).tobytes());q=dict(rollout_uid=c['rollout_uid'],first_action_bytes=first,positions_bytes=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),terminal=r['outcome']==old['outcome'],steps=r['episode_length']==old['episode_length'],new_x64=r['jax_enable_x64'],old_x64=old['jax_enable_x64'],reference_path=c['reference_path']);q['passed']=all(q[k] for k in ['first_action_bytes','positions_bytes','terminal','steps']);result[c['scene']].append(q)
 result={sc:dict(passed=len(v)==16 and all(x['passed'] for x in v),n=len(v),passed_count=sum(x['passed'] for x in v),checks=v) for sc,v in result.items()};write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),scenes=result,scope='Each scene independently gated; failed FP32 reproduction blocks prospective work only for that scene'));print({k:(v['passed'],v['passed_count']) for k,v in result.items()})
if __name__=='__main__':main()
