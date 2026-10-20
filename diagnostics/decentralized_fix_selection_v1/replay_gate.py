from support import *
G=guard()
def main():
 prereg();base=D/'decentralized_blackbox_planners_v1';p=read(base/'main_protocol.json');cases=read(base/'main_CASES.json');rows=[]
 # Two representatives in each planner/learning family, all relevant original arms.
 for name in ['DM0','DM5','G0','G5','AS0','AS2']:
  arms=['safety baseline','S500','RHR-fair'] if name not in ['AS0','AS2'] else ['F13','RHR-fair']
  rows.extend(next(c for c in cases if c['base_controller_name']==name and c['state_index']==0 and c['seed']==0 and c['arm']==a) for a in arms)
 assert len(rows)==16
 manifest=HERE/'replay_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]));pf=planner.preflight(manifest);write(HERE/'replay_PREFLIGHT.json',pf);assert pf['summary']['exact_reusable']==16;seal('replay_PREFLIGHT.receipt.json',[manifest,HERE/'replay_PREFLIGHT.json'],mandatory_reproduction_exception=True)
 cache={};results=[]
 for c in rows:
  rt=runtime(c['base_controller_name'],p,cache);s=p['states'][c['state_index']];env=rt.make_env(s);pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[];fw=fair(rt,c['rhr_rule']) if c['rhr_rule'] else None
  for t in range(rt.config.max_steps):
   key=key_for(s,c['seed'],t);eta=c['eta_a'] if t<c['switch_step'] else c['eta_b'];guard()
   a=fw.action(env,key,[0.,0.,0.])[0] if fw else decentralized_action(rt,env,key,eta,'local',np.zeros(4,int))[0]
   _,_,done,info=env.step(a);actions.append(np.asarray(a).copy());pos.append(env.positions.copy());vel.append(env.velocities.copy())
   if done:break
  path=base/'main/raw'/f"{c['rollout_uid']}.json";ref=read(path);a=dict(positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions));b=np.load(ref['trace_path']);checks={k:a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a};checks.update(outcome=info['termination']==ref['outcome'],steps=len(actions)==ref['physical_steps']);dest=HERE/'replay'/f"{c['rollout_uid']}.npz";dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**a)
  results.append(dict(case=c,reference=str(path),raw_sha256=sha(path),trace_sha256=sha(ref['trace_path']),checks=checks,passed=all(checks.values())));write(HERE/'REPLAY_GATE.json',dict(passed=all(x['passed'] for x in results),completed=len(results),required=16,results=results,guard=guard()));print('replay',len(results),checks,flush=True)
 source_inventory('imports_replay.json');seal('REPLAY_GATE.receipt.json',[HERE/'REPLAY_GATE.json'],passed=all(x['passed'] for x in results));assert all(x['passed'] for x in results)
if __name__=='__main__':main()
