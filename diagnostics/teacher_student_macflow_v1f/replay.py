from core import *
from data import teacher
from guard import guard
from episode_frozen import episode
from audit_utils import source_inventory
from shared_rollout_db.src.planner import preflight
G=guard()
def main():
 base=D/'decentralized_pipeline_v1';p=read(base/'main_protocol.json');names=p['controller_order'][:3]+[next(n for n in p['controller_order'] if 'local' in n)];rows=[c for c in read(base/'main_CASES.json') if c['base_controller_name'] in names and c['pipeline']=='local' and c['arm'] in ['F10','S500'] and c['state_index'] in [0,1] and c['seed']==0];assert len(rows)==16
 write(HERE/'gate_protocol.json',p);write(HERE/'REPLAY_CASES.json',rows);manifest=HERE/'REPLAY_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]));pf=preflight(manifest);assert pf['summary']['exact_reusable']==16;write(HERE/'REPLAY_CACHE_PREFLIGHT_CURRENT.json',pf)
 if not (HERE/'REPLAY_PREFLIGHT.receipt.json').exists():
  write(HERE/'REPLAY_CACHE_PREFLIGHT.json',pf);seal('REPLAY_PREFLIGHT.receipt.json',[manifest,HERE/'REPLAY_CACHE_PREFLIGHT.json',HERE/'REPLAY_CASES.json',HERE/'pipeline.py',HERE/'episode_frozen.py'],before_rollouts=True,mandatory_reproduction=True)
 cache={};results=[];ts=read(HERE/'TEACHERS.json')
 for c in rows:
  name=c['base_controller_name']
  if name not in cache:cache[name]=teacher(next(i for i,r in enumerate(ts) if r['name']==name))
  guard();r=episode(c,p,cache,'gate');ref=read(base/'main/raw'/f"{c['rollout_uid']}.json");a=np.load(r['trace_path']);b=np.load(ref['trace_path']);checks={k:a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in ['positions','velocities','actions']};checks.update(first_action=np.asarray(r['first_action'],np.float64).tobytes()==np.asarray(ref['first_action'],np.float64).tobytes(),outcome=r['outcome']==ref['outcome'],steps=r['episode_length']==ref['episode_length']);results.append(dict(case=c,checks=checks,passed=all(checks.values()),reference_sha256=sha(base/'main/raw'/f"{c['rollout_uid']}.json")));write(HERE/'ROUND28_REPLAY_GATE.json',dict(utc=utc(),passed=all(x['passed'] for x in results),completed=len(results),required=16,results=results,guard=guard()));assert all(checks.values()),checks;print('replay',len(results),'PASS',flush=True)
 source_inventory('imports_replay.json');seal('ROUND28_REPLAY_GATE.receipt.json',[HERE/'ROUND28_REPLAY_GATE.json',HERE/'pipeline.py',HERE/'episode_frozen.py'],before_student_rollouts=True)
if __name__=='__main__':main()
