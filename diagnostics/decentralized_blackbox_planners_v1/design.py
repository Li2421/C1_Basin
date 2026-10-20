from common import *
from registration import register,normalized_state,state_row,canonical,uid,eta_identity,connect
from audit_utils import source_inventory
ARMS=['nominal only','safety baseline','F10','F13','S300','S500','RHR-fair']
def create(phase):
 prereg();p=read(HERE/'CONTROLLERS.json');names=p['controller_order']
 if phase!='probe':assert (HERE/'CHOICES_FROZEN.receipt.json').exists()
 if phase=='main':names=read(HERE/'CAPABILITY_GATE.json')['eligible'];assert (HERE/'RHR_FROZEN.receipt.json').exists()
 if phase!='probe':
  names=[n for n in names if not n.startswith('DM') or read(HERE/'round33_REPLAY_GATE.json')['passed']]
 assert read(HERE/'round28_REPLAY_GATE.json')['passed']
 states=read(HERE/({'probe':'PROBE','pilot':'PILOT','tune':'TUNING','main':'EVAL'}[phase]+'_STATES.json'));eta=read(D/'decentralized_pipeline_v1/main_protocol.json')['eta'];p.update(states=states,controller_order=names,eta=eta,phase=phase,experiment_uid=EXP+'_'+phase)
 regs=[('experiment',dict(experiment_uid=p['experiment_uid'],name='Blackbox planners '+phase,path=str(HERE/phase),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(__file__),metadata_json=canonical(dict(preregistered=True,phase=phase))))]
 with connect(True) as db:
  for s in states:
   old=db.execute('SELECT * FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone()
   if old:assert json.loads(old['physical_state_json'])==s['physical']
   else:regs.append(state_row(s))
 cases=[]
 for name in names:
  if phase=='probe':plans=[('probeF10',None,10),('probeF13',None,13)]
  elif phase=='pilot':plans=[('safety baseline',None,None),('S500',None,13)]
  elif phase=='tune':plans=[(f'RHR_grid{i:02}',rule,None) for i,rule in enumerate(read(HERE/'RHR_GRID.json'))]
  else:plans=[(a,read(HERE/'RHR_SELECTED.json')[name.rstrip('0123456789')]['rule'] if a=='RHR-fair' else None,13 if a in ['F13','S300','S500'] else 10 if a=='F10' else None) for a in ARMS]
  for arm,rule,j in plans:
   v=p['profiles'][name]['payload'];ea=eta[j] if j is not None else [0.,0.,0.];eb=eta[10] if arm in ['S300','S500'] else ea;switch=int(arm[1:]) if arm in ['S300','S500'] else 700
   payload=dict(nominal=v,nominal_sources={n:sha(HERE/n) for n in ['controllers.py','legacy_controllers.py','gaussian.py','astar.py','runtime_new.py']},execution_source_sha256=sha(HERE/'worker.py'),pipeline_sha256=sha(HERE/'pipeline.py'),fair_rhr_sha256=sha(HERE/'fair_rhr.py') if rule else None,arm=arm,rule=rule,eta_schedule=dict(a=ea,b=eb,switch_step=switch),jax_enable_x64=False,rng='key_for(state.uid,seed,physical_step), root2026100403; DM fold_in(key,i); Gaussian joint key',probe=phase=='probe',terminate_on_success=phase!='probe',terminate_on_collision=phase!='probe',probe_steps=200 if phase=='probe' else None)
   ctl=uid('ctl',payload);eid,values,hx=eta_identity(ea);regs.extend([('controller_config',dict(controller_uid=ctl,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')),('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hx))])
   for si,s in enumerate(states):
    for seed in ([0,1] if phase=='main' else [0]):
     sk=canonical(dict(future_index=seed));rid=uid('probe' if phase=='probe' else 'roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk));cases.append(dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk,base_controller_name=name,family=name.rstrip('0123456789'),arm=arm,state_index=si,seed=seed,eta_a=ea,eta_b=eb,switch_step=switch,rhr_rule=rule,eta_index=j))
 write(HERE/f'{phase}_protocol.json',p);write(HERE/f'{phase}_CASES.json',cases);register(phase,regs)
 manifest=HERE/f'{phase}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in cases]));pf=planner.preflight(manifest);assert pf['summary']['ambiguous']==0;write(HERE/f'{phase}_CACHE_PREFLIGHT.json',pf);missing=[];reuse=[]
 with connect(True) as db:
  for c in cases:
   r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
   if r:assert not r['conflict_quarantined'];reuse.append(dict(case=c,row=dict(r)))
   else:missing.append(c)
 write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_REUSE_CASES.json',reuse);seal(f'{phase}_PREFLIGHT.receipt.json',[manifest,HERE/f'{phase}_CACHE_PREFLIGHT.json',HERE/f'{phase}_MISSING_CASES.json',HERE/f'{phase}_protocol.json',HERE/f'{phase}_CASES.json',HERE/'worker.py'],before_submission=True,missing=len(missing),reuse=len(reuse),guard=guard());source_inventory('imports_design_'+phase+'.json');print(phase,len(missing),flush=True)
if __name__=='__main__':create(sys.argv[1])
