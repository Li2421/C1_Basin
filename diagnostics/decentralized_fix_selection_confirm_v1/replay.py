from support import *
from episode_frozen import episode
G=guard()
def main():
 prereg();kind=sys.argv[1];rows=[];results=[];cache={}
 if kind=='round28':
  base=D/'decentralized_pipeline_v1';p=read(base/'main_protocol.json');names=p['controller_order'][:3]+[next(n for n in p['controller_order'] if 'local' in n)];rows=[c for c in read(base/'main_CASES.json') if c['base_controller_name'] in names and c['pipeline']=='local' and c['arm'] in ['F10','S500'] and c['state_index'] in [0,1] and c['seed']==0];write(HERE/'gate_protocol.json',p)
 elif kind=='round33':
  base=D/'axis2_controller_v1';p=read(base/'pilot_protocol.json');cases=read(base/'pilot_CASES.json');rows=[c for c in cases if c['base_controller_name'].startswith('DM') and c['state_index']==0];rows+=[c for c in cases if c['base_controller_name'] in ['DM0','DM5'] and c['state_index']==1]
 else:
  base=D/'decentralized_blackbox_planners_v1';p=read(base/'main_protocol.json');cases=read(base/'main_CASES.json')
  for name in ['DM0','DM5','G0','G5','AS0','AS2']:
   arms=['safety baseline','S500','RHR-fair'] if name not in ['AS0','AS2'] else ['F13','RHR-fair'];rows.extend(next(c for c in cases if c['base_controller_name']==name and c['state_index']==0 and c['seed']==0 and c['arm']==a) for a in arms)
 assert len(rows)==16;manifest=HERE/f'{kind}_REPLAY_PLANNED.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]));pf=planner.preflight(manifest);write(HERE/f'{kind}_REPLAY_PREFLIGHT.json',pf);assert pf['summary']['exact_reusable']==16;seal(f'{kind}_REPLAY_PREFLIGHT.receipt.json',[manifest,HERE/f'{kind}_REPLAY_PREFLIGHT.json'],mandatory_reproduction_exception=True)
 for c in rows:
  if kind=='round28':
   from poc.field_controller import Runtime
   n=c['base_controller_name'];v=p['base_profiles'][n]['payload']
   if n not in cache:cache[n]=Runtime('ring_exchange') if n=='base_u_v10_local_macflow' else RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
   guard();r=episode(c,p,cache,'gate');a=dict(np.load(r['trace_path']));outcome=r['outcome'];steps=r['physical_steps'];refpath=base/'main/raw'/f"{c['rollout_uid']}.json"
  else:
   rt=make(c['base_controller_name'],p,cache);s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);env=rt.make_env(s);pos=[env.positions.copy()];vel=[env.velocities.copy()];acts=[];fw=fair(rt,c['rhr_rule']) if c.get('rhr_rule') else None
   for t in range(rt.config.max_steps):
    guard();key=key_for(s,c['seed'],t);eta=c['eta_a'] if t<c['switch_step'] else c['eta_b']
    if kind=='round33':a=rt.action(env,key,np.asarray(eta),'old')[0]
    else:a=fw.action(env,key,[0.,0.,0.])[0] if fw else decentralized_action(rt,env,key,eta,'local',np.zeros(4,int))[0]
    _,_,done,info=env.step(a);acts.append(a.copy());pos.append(env.positions.copy());vel.append(env.velocities.copy())
    if done:break
   a=dict(positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(acts));outcome=info['termination'];steps=len(acts);refpath=base/('pilot' if kind=='round33' else 'main')/'raw'/f"{c['rollout_uid']}.json";dest=HERE/'replays'/kind/f"{c['rollout_uid']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**a)
  ref=read(refpath);b=np.load(ref['trace_path']);checks={k:a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in ['positions','velocities','actions']};checks.update(outcome=outcome==ref['outcome'],steps=steps==ref['physical_steps']);results.append(dict(case=c,checks=checks,passed=all(checks.values()),reference=str(refpath),raw_sha256=sha(refpath),trace_sha256=sha(ref['trace_path'])));write(HERE/f'{kind}_REPLAY_GATE.json',dict(passed=all(r['passed'] for r in results),completed=len(results),required=16,results=results,guard=guard()));print(kind,len(results),checks,flush=True)
 source_inventory(f'imports_replay_{kind}.json');seal(f'{kind}_REPLAY_GATE.receipt.json',[HERE/f'{kind}_REPLAY_GATE.json'],passed=all(r['passed'] for r in results));assert all(r['passed'] for r in results)
if __name__=='__main__':main()
