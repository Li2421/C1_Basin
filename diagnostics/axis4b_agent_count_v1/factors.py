from arm_binding import *

def build(stage,kind=None,Ns=None):
 p=protocol();rows=[];states=read(HERE/'STATES.json');cfg=read(HERE/'DESIGN_CONFIG.json');gate=read(HERE/'CONSISTENCY_GATE.json');elig=read(HERE/'CAPABILITY_GATE.json') if (HERE/'CAPABILITY_GATE.json').exists() else None
 for n in (cfg['N'] if Ns is None else Ns):
  ss=states[str(n)]['eval' if stage!='pilot' else 'pilot']
  for s in ss:
   sid=s['scenario_uid'];base=s['evaluation_config'];p['scenarios'][sid]=dict(scenario_uid=sid,name=f'Gap1_N{n}',code_config_fingerprint=uid('cfg',base),metadata_json=canonical(dict(config=base,environment_sha256=sha(ROOT/'bottleneck_family/environment.py'))));p['states'].append(s)
   if kind=='expert':
    if gate['expert_passed']:rows.append(case(p,s,'expert','expert'))
    continue
   if not gate['primitive_passed']:continue
   for ctl in ['Flow','P2']:
    if stage!='pilot' and elig['controllers'][str(n)][ctl]['status']!='PASS':continue
    arms=['safety_baseline','F10','S13t2'] if stage=='pilot' else cfg['arms']
    for arm in arms:
     for seed in [0] if stage=='pilot' else [0,1]:rows.append(case(p,s,ctl,arm,seed))
 if not rows:write(HERE/f'{stage}_NO_REQUESTS.json',dict(utc=utc(),reason='No eligible/gated controller or reference'));return
 # Deterministic seed0 before its exact seed1 derivation; stochasticFlow still runsboth.
 rows.sort(key=lambda c:(c['seed'],c['N'],c['state_index'],c['kind'],cfg['arms'].index(c['arm']) if c['arm'] in cfg['arms'] else 0));lookup={(c['N'],c['kind'],c['state_uid'],c['arm'],c['seed']):c for c in rows}
 for c in rows:
  if c['kind']=='P2' and c['seed']==1:c['deterministic_source_case']=dict(phase=stage,rollout_uid=lookup[c['N'],'P2',c['state_uid'],c['arm'],0]['rollout_uid'])
  if stage in ['source','target'] and c['arm'].startswith(('S13','S15')) and not (c['kind']=='P2' and c['seed']==1):
   parent=lookup[c['N'],c['kind'],c['state_uid'],'F13' if c['arm'].startswith('S13') else 'F15',c['seed']]
   c['shared_prefix_source_case']=dict(phase=stage,rollout_uid=parent['rollout_uid'])
 register(stage,p,rows,{v['controller_uid']:v for v in p['profiles'].values()})
def capability():
 rows=read(HERE/'pilot_CASES.json') if (HERE/'pilot_CASES.json').exists() else [];cfg=read(HERE/'DESIGN_CONFIG.json');out={}
 for n in cfg['N']:
  out[str(n)]={}
  for kind in ['Flow','P2']:
   rates={}
   for arm in ['safety_baseline','F10','S13t2']:
    rr=[read(HERE/'pilot/raw'/f"{c['rollout_uid']}.json") for c in rows if c['N']==n and c['kind']==kind and c['arm']==arm]
    if len(rr)==16:rates[arm]=[sum(q['success'] for q in rr)/16,sum(q['success']+q['numerical_failure'] for q in rr)/16]
   bounds=np.max(list(rates.values()),axis=0).tolist() if rates else [None,None];status='GAP' if not rates else 'PASS' if bounds[0]>=.2 else 'FAIL_CONTROLLER_CAPABILITY' if bounds[1]<.2 else 'INCONCLUSIVE';out[str(n)][kind]=dict(status=status,max_success_bounds=bounds,arms=rates)
 source=next((n for n in cfg['N'] if any(v['status']=='PASS' for v in out[str(n)].values())),None);write(HERE/'CAPABILITY_GATE.json',dict(utc=utc(),controllers=out,source_N=source,threshold=.2,reciprocal='GAP historical first-action byte gate unavailable'));seal('CAPABILITY_GATE_FROZEN.receipt.json',[HERE/'CAPABILITY_GATE.json'],before_source_and_target_main=True);print('Capability/source',source,{n:{k:v['status'] for k,v in q.items()} for n,q in out.items()},flush=True)
if __name__=='__main__':
 if sys.argv[1]=='capability':capability()
 elif sys.argv[1]=='pilot':build('pilot')
 elif sys.argv[1]=='expert':build('expert',kind='expert')
 elif sys.argv[1]=='source':build('source',Ns=[read(HERE/'CAPABILITY_GATE.json')['source_N']])
 elif sys.argv[1]=='target':build('target',Ns=[n for n in read(HERE/'DESIGN_CONFIG.json')['N'] if n>read(HERE/'CAPABILITY_GATE.json')['source_N']])
