from common import *
checks=[]
for c in read(HERE/'gate_CASES.json'):
 new=read(HERE/'gate/raw'/f"{c['rollout_uid']}.json");path=D/'controller_universal_eta_schedule_v1/main/raw'/f"{c['reference_rollout_uid']}.json";old=read(path)
 a=np.load(new['trace_path'])['positions'];b=np.load(old['trace_path'])['positions']
 first=np.asarray(new['first_action'],np.float64).tobytes()==np.asarray(old['first_action'],np.float64).tobytes();positions=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
 check=dict(rollout_uid=c['rollout_uid'],reference=str(path),first_action_bytes=first,position_trajectory_bytes=positions,outcome=new['outcome']==old['outcome'],episode_length=new['episode_length']==old['episode_length'],old_trace_sha256=sha(old['trace_path']),new_trace_sha256=sha(new['trace_path']))
 check['passed']=all(check[k] for k in ['first_action_bytes','position_trajectory_bytes','outcome','episode_length']);checks.append(check)
write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),passed=len(checks)==16 and all(c['passed'] for c in checks),checks=checks));print('Gate',sum(c['passed'] for c in checks),'/16')
