from support import *
from registration import register,state_row,canonical,uid,eta_identity,connect
G=guard()
def main():
 prereg();assert (HERE/'ALL_CHOICES_FROZEN.receipt.json').exists();
 import time
 while not (HERE/'ANALYSIS_CODE_FROZEN.receipt.json').exists():time.sleep(10)
 p=read(HERE/'CONTROLLERS.json');p['states']=read(HERE/'EVAL_STATES.json');p['experiment_uid']=EXP+'_main';choices=read(HERE/'NEW_INSTANCE_CHOICES.json')['controllers'];regs=[('experiment',dict(experiment_uid=p['experiment_uid'],name='Prospective frozen fix selection confirmation',path=str(HERE/'main'),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(__file__),metadata_json=canonical(dict(confirmatory=True,primary_rule=read(HERE/'FROZEN_RULES.json')['primary']))))]
 with connect(True) as db:
  for s in p['states']:
   old=db.execute('SELECT * FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone()
   if old:assert json.loads(old['physical_state_json'])==s['physical']
   else:regs.append(state_row(s))
 cases=[]
 for n in p['controller_order']:
  v=p['profiles'][n]['payload'];plans=ARMS.copy();extra=choices[n]['old_critic']
  if extra not in plans:plans.append(extra)
  for arm in plans:
   j=13 if arm in ['F13','S300','S500'] else 10 if arm=='F10' else None;ea=p['eta'][j] if j is not None else [0.,0.,0.];eb=p['eta'][10] if arm in ['S300','S500'] else ea;switch=int(arm[1:]) if arm in ['S300','S500'] else 700;rule=RULES.get(arm);payload=dict(nominal=v,arm=arm,rule=rule,eta_schedule=dict(a=ea,b=eb,switch_step=switch),execution_source_sha256=sha(HERE/'worker.py'),pipeline_sha256=sha(HERE/'pipeline.py'),sources={f:sha(HERE/f) for f in ['support.py','runtime_new.py','student.py','data.py','controllers.py','legacy_controllers.py','gaussian.py','astar.py','fair_rhr.py']},jax_enable_x64=False,rng='Original key_for(state.uid,seed,step); original Student private or shared fold-in stream',native_success_and_termination_unchanged=True);ctl=uid('ctl',payload);eid,values,hx=eta_identity(ea);regs.extend([('controller_config',dict(controller_uid=ctl,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')),('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hx))])
   for si,s in enumerate(p['states']):
    for seed in [0,1]:
     sk=canonical(dict(future_index=seed));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk));cases.append(dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk,base_controller_name=n,family=next(f for f in FAMILIES if n.startswith(f)),arm=arm,state_index=si,seed=seed,eta_a=ea,eta_b=eb,switch_step=switch,rhr_rule=rule,eta_index=j,primary_six_arm=arm in ARMS))
 write(HERE/'main_protocol.json',p);write(HERE/'main_CASES.json',cases);register('main',regs);manifest=HERE/'main_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in cases]));pf=planner.preflight(manifest);write(HERE/'main_CACHE_PREFLIGHT.json',pf);assert pf['summary']['ambiguous']==0;missing=[];reuse=[]
 with connect(True) as db:
  for c in cases:
   row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
   if row:reuse.append(dict(case=c,row=dict(row)))
   else:missing.append(c)
 write(HERE/'main_MISSING_CASES.json',missing);write(HERE/'main_REUSE_CASES.json',reuse);seal('main_PREFLIGHT.receipt.json',[HERE/n for n in ['main_protocol.json','main_CASES.json','main_PLANNED_ROLLOUTS.json','main_CACHE_PREFLIGHT.json','main_MISSING_CASES.json','worker.py','run.py','support.py','PREREGISTRATION.md','ALL_CHOICES_FROZEN.receipt.json']],missing=len(missing),reuse=len(reuse),before_submission=True,guard=guard());source_inventory('imports_design.json');print('TASK PREFLIGHT',len(cases),'six-arm',58*64*6,'missing',len(missing),'reuse',len(reuse),flush=True)
if __name__=='__main__':main()
