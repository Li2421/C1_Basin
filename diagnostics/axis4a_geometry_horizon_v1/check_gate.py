from common import *
def main(phase='gate'):
 checks=[]
 for c in read(HERE/f'{phase}_CASES.json'):
  r=read(HERE/phase/'raw'/f"{c['rollout_uid']}.json");old=read(c['reference_path']);assert sha(c['reference_path'])==c['reference_sha256'];a=np.load(r['trace_path'])['positions'];z=np.load(old['trace_path']);b=z['positions'];oldfirst=old.get('first_action')
  if oldfirst is None and 'actions' in z and len(z['actions']):oldfirst=z['actions'][0]
  first=(r['first_action'] is None and oldfirst is None) or (r['first_action'] is not None and oldfirst is not None and np.asarray(r['first_action'],np.float64).tobytes()==np.asarray(oldfirst,np.float64).tobytes());q=dict(rollout_uid=c['rollout_uid'],first_action_bytes=first,positions_bytes=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),terminal=r['outcome']==old['outcome'],steps=r['episode_length']==old['episode_length'],new_x64=r['jax_enable_x64'],old_x64=old['jax_enable_x64'],reference_path=c['reference_path']);q['passed']=all(q[k] for k in ['first_action_bytes','positions_bytes','terminal','steps']);checks.append(q)
 result=dict(utc=utc(),phase=phase,passed=len(checks)==16 and all(x['passed'] for x in checks),passed_count=sum(x['passed'] for x in checks),checks=checks);write(HERE/('CONSISTENCY_GATE.json' if phase=='gate' else 'EXPERT_CONSISTENCY_GATE.json'),result);print(phase,result['passed'],result['passed_count'])
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'gate')
