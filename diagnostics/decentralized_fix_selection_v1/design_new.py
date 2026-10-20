from support import *
from registration import register,canonical,uid,eta_identity,connect,state_row
G=guard()
def main():
 prereg();p=read(HERE/'CONTROLLERS.json');names=p['controller_order'];old={};history=[]
 for r in read(D/'teacher_student_line_closeout_v1/MANIFEST.json'):
  if r['group'] not in ['V0','TD']:continue
  arm='RHR-C' if r['group']=='V0' and r['arm']=='RHR-fair' else r['arm']
  if arm not in ARMS:continue
  name=('ST' if r['group']=='V0' else 'TD')+f"{r['controller_index']:02}";rr=dict(r,name=name,arm=arm,source_round=r['round']);old[name,arm,r['state_index'],r['seed']]=rr
 for r in read(D/'decentralized_blackbox_planners_v1/ANALYSIS_MANIFEST.json'):
  if r['group'] not in ['DM','G','AS']:continue
  arm=r['arm']
  if arm=='RHR-fair':
   if r['group']=='G':continue # progress/40/.10 is outside the frozen six-arm menu.
   arm='RHR-A' if r['group']=='DM' else 'RHR-B'
  if arm not in ARMS:continue
  rr=dict(r,arm=arm,source_round='decentralized_blackbox_planners_v1');old[r['name'],arm,r['state_index'],r['seed']]=rr
 regs=[('experiment',dict(experiment_uid=p['experiment_uid'],name='DEVELOPMENT decentralized fix selection supplements',path=str(HERE/'supplement'),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(__file__),metadata_json=canonical(dict(families=FAMILIES,preregistered=True))))];cases=[];reuse=[];manifest=[]
 for name in names:
  family=next(f for f in FAMILIES if name.startswith(f));v=p['profiles'][name]['payload']
  for arm in ARMS:
   ea=p['eta'][13] if arm in ['F13','S500'] else [0.,0.,0.];eb=p['eta'][10] if arm=='S500' else ea;switch=500 if arm=='S500' else 700;rule=RULES.get(arm);eid,values,hx=eta_identity(ea)
   payload=dict(nominal=v,execution_source_sha256=sha(HERE/'worker.py'),pipeline_sha256=sha(HERE/'pipeline.py'),runtime_sources={n:sha(HERE/n) for n in ['runtime_new.py','student.py','data.py','controllers.py','legacy_controllers.py','gaussian.py','astar.py','fair_rhr.py']},arm=arm,rule=rule,eta_schedule=dict(a=ea,b=eb,switch_step=switch),jax_enable_x64=False,rng='unchanged key_for(original state RNG alias, seed, physical step)',termination='native task rules')
   ctl=uid('ctl',payload);newcells=[]
   for si,s in enumerate(p['states']):
    for seed in [0,1]:
     key=(name,arm,si,seed)
     if key in old:
      r=old[key];raw=read(r['raw_path']);assert sha(r['raw_path'])==r['raw_sha256'] and sha(r['trace_path'])==r['trace_sha256'];c=raw['case'];assert c['seed']==seed and c['state_index']==si
      assert np.array_equal(np.asarray(raw['initial_positions']),np.asarray(s['physical']['positions'])) if 'initial_positions' in raw else True
      manifest.append({k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']});reuse.append(r);continue
     sk=canonical(dict(future_index=seed));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk));c=dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk,base_controller_name=name,family=family,arm=arm,state_index=si,seed=seed,eta_a=ea,eta_b=eb,switch_step=switch,rhr_rule=rule,eta_index=13 if arm in ['F13','S500'] else None);cases.append(c);newcells.append(c);manifest.append({k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']})
   if newcells:regs.extend([('controller_config',dict(controller_uid=ctl,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=v.get('flow_checkpoint_sha256'),config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')),('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hx))])
 write(HERE/'HISTORY_REUSE.json',reuse);write(HERE/'HISTORY_AUDIT.json',dict(reused=len(reuse),missing=len(cases),total=len(names)*6*64,compatibility='Frozen checkpoints, original state and RNG aliases, exact historical controller identity and original registered UIDs. Right-hand rules matched by full mode/window/speed/distance, not display label.',diagnostic_gap='Existing R5 has different actions/horizon; task traces omit per-step nominal. New 300-step termination-disabled safety-only diagnostics supply all seven preregistered unlabeled features.',excluded_different_rule='Gaussian round30e RHR-fair uses progress/40/.10, not menu RHR-C progress/40/.25',source_manifests={str(q):sha(q) for q in [D/'teacher_student_line_closeout_v1/MANIFEST.json',D/'decentralized_blackbox_planners_v1/ANALYSIS_MANIFEST.json']}))
 write(HERE/'supplement_protocol.json',p);write(HERE/'supplement_CASES.json',cases);register('supplement',regs)
 path=HERE/'ALL_PLANNED_ROLLOUTS.json';write(path,dict(requests=manifest));pf=planner.preflight(path);write(HERE/'ALL_CACHE_PREFLIGHT.json',pf);assert pf['summary']['ambiguous']==0
 # Existing numerical unknowns, if any, are retained and never sent to workers.
 with connect(True) as db:
  for r in reuse:
   row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone();assert row and not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE';assert bool(row['numerical_failure'])==bool(r['unknown'])
  missing=[];already=[]
  for c in cases:
   row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
   if row:already.append(dict(case=c,row=dict(row)))
   else:missing.append(c)
 write(HERE/'supplement_MISSING_CASES.json',missing);write(HERE/'supplement_REUSE_CASES.json',already);write(HERE/'supplement_PLANNED_ROLLOUTS.json',dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in cases]));seal('supplement_PREFLIGHT.receipt.json',[path,HERE/'ALL_CACHE_PREFLIGHT.json',HERE/'supplement_MISSING_CASES.json',HERE/'supplement_protocol.json',HERE/'worker.py',HERE/'run.py',HERE/'support.py',HERE/'PREREGISTRATION.md'],missing=len(missing),historical_reuse=len(reuse),guard=guard())
 d=dict(p,states=read(HERE/'PROBE_STATES.json'));write(HERE/'diagnostic_protocol.json',d);dc=[dict(base_controller_name=n,state_index=i,seed=0,arm='safety baseline',rollout_uid=f'diagnostic_{n}_{i:02}') for n in names for i in range(16)];write(HERE/'diagnostic_MISSING_CASES.json',dc);seal('diagnostic_PREFLIGHT.receipt.json',[HERE/'diagnostic_protocol.json',HERE/'diagnostic_MISSING_CASES.json',HERE/'run.py',HERE/'support.py',HERE/'PREREGISTRATION.md'],unlabeled=True,task_db_requests=0,guard=guard());source_inventory('imports_design.json');print('reuse/missing/diagnostics',len(reuse),len(missing),len(dc),pf['summary'],flush=True)
if __name__=='__main__':main()
