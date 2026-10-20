from core import *
from runtime import execute,jax
from shared_rollout_db.src import cache_writer
cache_writer.ROOT=HERE
def gate():
 prereg();p=read(HERE/'protocol.json');checks=[];cache={}
 for c in read(HERE/'gate_CASES.json'):
  r=execute(c,p,cache,'gate');old=read(OLD/'main/raw'/f"{c['rollout_uid']}.json");z=np.load(r['trace_path']);ref=np.load(old['trace_path']);a=z['actions'];b=ref['actions']
  first=bool(len(a) and len(b) and a[0].dtype==b[0].dtype and a[0].shape==b[0].shape and a[0].tobytes()==b[0].tobytes())
  pos=z['positions'];rpos=ref['positions'];full=pos.shape==rpos.shape and pos.dtype==rpos.dtype and pos.tobytes()==rpos.tobytes()
  ok=first and r['outcome']==old['outcome'] and r['episode_length']==old['episode_length']
  check=dict(controller=c['base_controller_name'],arm=c['arm'],rollout_uid=c['rollout_uid'],termination_equal=r['outcome']==old['outcome'],steps_equal=r['episode_length']==old['episode_length'],first_action_bitwise_equal=first,full_positions_bitwise_equal=full,passed=ok,actual_outcome=r['outcome'],expected_outcome=old['outcome'],actual_steps=r['episode_length'],expected_steps=old['episode_length'],jax_enable_x64=r['jax_enable_x64'])
  write(HERE/'gate/raw'/f"{c['rollout_uid']}.json",r);checks.append(check);write(HERE/'CONSISTENCY_GATE.json',dict(utc=utc(),passed=len(checks)==8 and all(x['passed'] for x in checks),checks=checks,completed=len(checks),required=8))
  print(json.dumps(check),flush=True)
  if not ok:
   (HERE/'REPORT.md').write_text('一致性门失败；未启动 baseline rollout。不能给出三类成功率提升与 CI。\n\n详见 CONSISTENCY_GATE.json；按 prompt 停止，保留所有证据。\n')
   (HERE/'HANDOFF.md').write_text('Round29 stopped at consistency gate. Read PREREGISTRATION.md, FROZEN_INPUTS.json and CONSISTENCY_GATE.json. No baseline executed. Do not continue without resolving mismatch under new authorization.\n')
   raise SystemExit('CONSISTENCY GATE FAILED; STOP')
def worker(index,count):
 prereg();assert read(HERE/'CONSISTENCY_GATE.json')['passed'];p=read(HERE/'protocol.json');cache={};rows=read(HERE/'main_MISSING_CASES.json')[index::count];new=0
 for c in rows:
  out=HERE/'main/raw'/f"{c['rollout_uid']}.json"
  if out.exists():assert read(out)['case']==c;continue
  r=execute(c,p,cache,'main');write(out,r);cache_writer.append_journal([r],EXP,f'main_{index}');new+=1
  print(json.dumps(dict(worker=index,new=new,total=len(rows),controller=c['base_controller_name'],chain=c['chain'],seconds=r['elapsed_seconds'],outcome=r['outcome'])),flush=True)
 write(HERE/'main'/f'worker_{index}.json',dict(utc=utc(),complete=True,assigned=len(rows),new=new,jax_enable_x64=bool(jax.config.x64_enabled)))
if __name__=='__main__':
 if sys.argv[1]=='gate':gate()
 else:worker(int(sys.argv[2]),int(sys.argv[3]))
