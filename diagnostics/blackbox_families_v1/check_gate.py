from common import *
checks=[]
for c in read(HERE/'gate_CASES.json'):
 r=read(HERE/'gate/raw'/f"{c['rollout_uid']}.json");old=read(c['reference_path']);assert sha(c['reference_path'])==c['reference_sha256']
 with np.load(r['trace_path']) as z:a=z['positions']
 with np.load(old['trace_path']) as z:b=z['positions']
 v=dict(group=c['cohort'],reference=c['reference_path'],first_action_bytes=np.asarray(r['first_action'],np.float64).tobytes()==np.asarray(old['first_action'],np.float64).tobytes(),positions_bytes=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),outcome=r['outcome']==old['outcome'],steps=r['episode_length']==old['episode_length']);v['passed']=all(v[k] for k in ['first_action_bytes','positions_bytes','outcome','steps']);checks.append(v)
groups={g:dict(count=sum(x['group']==g for x in checks),passed=all(x['passed'] for x in checks if x['group']==g)) for g in sorted({x['group'] for x in checks})};write(HERE/'REPRODUCTION_GATE.json',dict(utc=utc(),passed=len(checks)==32 and all(x['passed'] for x in checks),groups=groups,checks=checks));print(groups)
