from common import *
from registry import register,uid,canonical,eta_identity
import copy
PARAMS={'DM1':[10,64,100],'DM3':[10,256,100]}
ARMS=['nominal_only','safety_baseline','F10','F13','F15','S13t1','S13t2','S15t1','S15t2']
def protocol():return dict(states=[],profiles={},scenarios={},eta=read(R36/'pilot_protocol.json')['eta'])
def addstate(p,s):
 if s['state_uid'] not in {x['state_uid'] for x in p['states']}:p['states'].append(copy.deepcopy(s))
def case(p,s,name,arm,seed=0,timing=False):
 addstate(p,s);H=s['evaluation_config']['max_steps'];v=dict(scenario_uid=s['scenario_uid'],controller_class='distributed_shooting',parameters=PARAMS[name],horizon=H,dt=s['evaluation_config']['dt'],success_semantics='native all_agents_reach_and_no_swept_collision; no diagnostic deadlock terminal',conditioning=dict(chain='bound_goal_stop_projection_orthoflow3_projection',arm=arm,timing_only=timing),rng=dict(root=2026093011,scheme='state_uid_sha32/future/physical_step/agent'),orthoflow3_sha256=sha(ROOT/'shared_control/basis_families.py'),safety_projection_sha256=sha(ROOT/'shared_control/hard_projection.py'),cbf=read(R36/'pilot_protocol.json')['profiles'][next(iter(read(R36/'pilot_protocol.json')['profiles']))]['payload']['cbf'],execution_source_sha256={f:sha(HERE/f) for f in ['gap_mpc.py','mpc_general.py','gap_controllers.py']},round37_engine='gap_mpc');key=uid('profile',v);cid=uid('ctl',v);p['profiles'][key]=dict(controller_uid=cid,payload=v);a=[0.,0.,0.] if arm in ['nominal_only','safety_baseline'] else p['eta'][int(arm[1:])] if arm.startswith('F') else p['eta'][13 if arm.startswith('S13') else 15];b=p['eta'][10] if arm.startswith('S') else a;sw=round((3 if arm.endswith('t1') else 5)*H/7) if arm.startswith('S') else H;eid=eta_identity(a)[0];sk=canonical(dict(future_index=seed));return dict(N=s['N'],H=H,kind='DM',base_controller_name=name,parameters=PARAMS[name],arm=arm,engine='timing' if timing else 'gap_mpc',state_uid=s['state_uid'],state_index=s['study_index'],profile_key=key,controller_uid=cid,eta_uid=eid,seed_key=sk,rollout_uid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk)),seed=seed,eta_a=a,eta_b=b,switch_step=sw,evaluation_config=s['actual_config'])
def replay():
 p=protocol();oldp=read(R36/'pilot_protocol.json');cases=read(R36/'pilot_CASES.json');rows=[]
 for n,k in [(2,6),(10,5),(20,5)]:
  for oc in sorted([c for c in cases if c['N']==n and c['kind']=='Flow'],key=lambda c:(c['state_uid'],c['arm']))[:k]:
   s=next(s for s in oldp['states'] if s['state_uid']==oc['state_uid']);addstate(p,s);c=copy.deepcopy(oc);v=copy.deepcopy(oldp['profiles'][c['profile_key']]['payload']);v.update(validation='round37_full_TT_replay',execution_source_sha256={f:sha(HERE/f) for f in ['gap_runtime.py','gap_controllers.py']});key=uid('profile',v);cid=uid('ctl',v);p['profiles'][key]=dict(controller_uid=cid,payload=v);c.update(profile_key=key,controller_uid=cid,engine='gap_historical');c['rollout_uid']=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=cid,seed=c['seed_key']));path=R36/'pilot/raw'/f"{oc['rollout_uid']}.json";c.update(reference_path=str(path),reference_sha256=sha(path));rows.append(c)
 register('replay_E',p,rows)
def timing():
 p=protocol();rows=[];states=read(R36/'STATES.json')
 for n in [2,10,20]:
  for name in PARAMS:
   for s in states[str(n)]['pilot'][:2]:rows.append(case(p,s,name,'safety_baseline',timing=True))
 register('gap_timing',p,rows)
def gate():
 p=protocol();rows=[];states=read(R36/'STATES.json');budget=read(HERE/'MPC_TIMING_REPORT.json')
 for n in [2,10,20]:
  if not budget[str(n)]['eligible']:continue
  for name in PARAMS:
   for s in states[str(n)]['pilot']:
    for arm in ['safety_baseline','F10','S13t2']:rows.append(case(p,s,name,arm))
 register('gap_capability',p,rows)
def mainphase(n,which):
 p=protocol();rows=[];states=read(R36/'STATES.json')[str(n)]['eval'];gate=read(HERE/'MPC_CAPABILITY_GATE.json')[str(n)];menus=ARMS[:5] if which=='fixed' else ARMS[5:]
 for name in PARAMS:
  if gate[name]['status']!='PASS':continue
  for s in states:
   for seed in [0,1]:
    for arm in menus:
     c=case(p,s,name,arm,seed)
     if which=='schedules' and read(HERE/'GAP_EXACT_PREFIX_PROOF.json')['passed']:
      parent=case(p,s,name,'F13' if arm.startswith('S13') else 'F15',seed);path=HERE/f'gap_N{n}_fixed/raw'/f"{parent['rollout_uid']}.json";assert path.exists();c['prefix_source']=dict(raw_path=str(path),raw_sha256=sha(path))
     rows.append(c)
 register(f'gap_N{n}_{which}',p,rows)
if __name__=='__main__':
 if sys.argv[1]=='mainphase':mainphase(int(sys.argv[2]),sys.argv[3])
 else:globals()[sys.argv[1]]()
