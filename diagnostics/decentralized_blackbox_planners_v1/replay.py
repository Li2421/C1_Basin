from runtime_new import *
from episode_frozen import episode
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
from shared_rollout_db.src.planner import preflight
from audit_utils import source_inventory
import jax
G=guard()
def compare(c,a,b,path):
 checks={k:a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in ['positions','velocities','actions']};return dict(case=c,checks=checks,passed=all(checks.values()),reference=str(path),reference_sha256=sha(path))
def main():
 prereg();assert read(HERE/'UNIT_TESTS.json')['passed'];kind=sys.argv[1];results=[]
 if kind=='round28':
  base=D/'decentralized_pipeline_v1';p=read(base/'main_protocol.json');names=p['controller_order'][:3]+[next(n for n in p['controller_order'] if 'local' in n)];rows=[c for c in read(base/'main_CASES.json') if c['base_controller_name'] in names and c['pipeline']=='local' and c['arm'] in ['F10','S500'] and c['state_index'] in [0,1] and c['seed']==0];write(HERE/'gate_protocol.json',p)
 else:
  base=D/'axis2_controller_v1';p=read(base/'pilot_protocol.json');rows=[c for c in read(base/'pilot_CASES.json') if c['base_controller_name'].startswith('DM') and c['state_index']==0];rows += [c for c in read(base/'pilot_CASES.json') if c['base_controller_name'] in ['DM0','DM5'] and c['state_index']==1];assert len(rows)==16
 manifest=HERE/f'{kind}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in rows]));pf=preflight(manifest);write(HERE/f'{kind}_PREFLIGHT.json',pf);assert pf['summary']['exact_reusable']==16;seal(f'{kind}_PREFLIGHT.receipt.json',[manifest,HERE/f'{kind}_PREFLIGHT.json'],mandatory_replay=True)
 cache={}
 for c in rows:
  if kind=='round28':
   from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
   from poc.field_controller import Runtime
   name=c['base_controller_name'];v=p['base_profiles'][name]['payload']
   if name not in cache:cache[name]=Runtime('ring_exchange') if name=='base_u_v10_local_macflow' else RingFlowAdapter(v['flow_checkpoint_path'],read(v['adapter_manifest_path']),v['observation_frame'])
   guard();r=episode(c,p,cache,'gate');a=dict(np.load(r['trace_path']));refpath=base/'main/raw'/f"{c['rollout_uid']}.json";ref=read(refpath);b=dict(np.load(ref['trace_path']));outcome=r['outcome'];steps=r['physical_steps']
  else:
   rt=make(c['base_controller_name'],p,cache);s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);env=rt.make_env(s);pos=[env.positions.copy()];vel=[env.velocities.copy()];actions=[]
   for t in range(rt.config.max_steps):
    key=key_for(s,c['seed'],t);eta=c['eta_a'] if t<c['switch_step'] else c['eta_b'];action,_,_=rt.action(env,key,np.asarray(eta),'old');_,_,done,info=env.step(action);actions.append(action.copy());pos.append(env.positions.copy());vel.append(env.velocities.copy())
    if done:break
   a=dict(positions=np.asarray(pos),velocities=np.asarray(vel),actions=np.asarray(actions));refpath=base/'pilot/raw'/f"{c['rollout_uid']}.json";ref=read(refpath);b=dict(np.load(ref['trace_path']));outcome=info['termination'];steps=len(actions)
   dest=HERE/'round33_replay'/f"{c['rollout_uid']}.npz";dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**a)
  r=compare(c,a,b,refpath);r['checks'].update(outcome=outcome==ref['outcome'],steps=steps==ref['physical_steps']);r['passed']=all(r['checks'].values());results.append(r);write(HERE/f'{kind}_REPLAY_GATE.json',dict(passed=all(x['passed'] for x in results),completed=len(results),required=16,results=results,guard=guard()));print(kind,len(results),r['passed'],flush=True)
 source_inventory('imports_'+kind+'_replay.json');seal(kind+'_REPLAY_GATE.receipt.json',[HERE/f'{kind}_REPLAY_GATE.json'],passed=all(x['passed'] for x in results));assert all(x['passed'] for x in results)
if __name__=='__main__':main()
