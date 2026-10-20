from common import *
from audit_utils import source_inventory

def replay():
 checks=[]
 for c in read(HERE/'replay_E_import_repair_CASES.json'):
  r=read(HERE/'replay_E_import_repair/raw'/f"{c['rollout_uid']}.json");old=read(c['reference_path']);assert sha(c['reference_path'])==c['reference_sha256'];a=np.load(r['trace_path'])['positions'];b=np.load(old['trace_path'])['positions'];fa=r['first_action'];fb=old['first_action'];q=dict(N=c['N'],arm=c['arm'],first_action=(fa is None and fb is None) or (fa is not None and fb is not None and np.asarray(fa,np.float64).tobytes()==np.asarray(fb,np.float64).tobytes()),positions=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),termination=r['outcome']==old['outcome'],steps=r['episode_length']==old['episode_length']);q['passed']=all(q[k] for k in ['first_action','positions','termination','steps']);checks.append(q)
 write(HERE/'E_REPRODUCTION_GATE.json',dict(utc=utc(),passed=all(q['passed'] for q in checks),checks=checks));seal('E_REPRODUCTION_GATE.receipt.json',[HERE/'E_REPRODUCTION_GATE.json']);source_inventory('imports_check_E.json')

def prefix():
 from gap_mpc_v2 import BottleneckEnv,Config,restore,action_for,rootkey,CertifiedHardSafetyFilter,HardProjectionConfig,get_basis_family
 checks=[];p=read(HERE/'gap_timing_protocol.json')
 for c in read(HERE/'gap_timing_CASES.json'):
  r=read(HERE/'gap_timing/raw'/f"{c['rollout_uid']}.json");z=np.load(r['trace_path']);s=next(s for s in p['states'] if s['state_uid']==c['state_uid']);env=BottleneckEnv(Config(**c['evaluation_config']));env.reset(np.asarray(s['physical']['positions']),np.asarray(s['physical']['goals']))
  for t in [20,100]:
   if len(z['actions'])<=t:continue
   restore(env,z['positions'][:t+1],z['velocities'][:t+1]);a=action_for(env,rootkey(s,c['seed']),t,c,CertifiedHardSafetyFilter(HardProjectionConfig()),get_basis_family('orthoflow3'));b=z['actions'][t];passed=a.dtype==b.dtype and a.tobytes()==b.tobytes();assert env.step_count==t and np.array_equal(env.goals,z['goals']) and np.array_equal(env.walls,z['walls']) and not env.done and not env.collided;checks.append(dict(N=c['N'],parameters=c['parameters'],state_index=c['state_index'],step=t,next_action_bitwise=passed,snapshot_complete=True))
 out=dict(utc=utc(),passed=len(checks)==24 and all(q['next_action_bitwise'] for q in checks),checks=checks,new_physical_steps=0,proof='Native next state/event depends only on snapshot/action; restored all dynamic fields, full history, original goals/walls/config and stateless step/agent random key. Stateless safety projection recreated. Exact parent prefix has identical parameters, state identity, future, config, source and early eta. Terminal-before-switch labels copied; unknown-before-divergence preserved.');write(HERE/'GAP_EXACT_PREFIX_PROOF.json',out);seal('GAP_EXACT_PREFIX_PROOF.receipt.json',[HERE/'GAP_EXACT_PREFIX_PROOF.json',HERE/'gap_mpc.py']);source_inventory('imports_check_gap_prefix.json')
if __name__=='__main__':globals()[sys.argv[1]]()
