from core import *
from guard import guard
from registration import register,connect,canonical,uid,eta_identity,state_row
from shared_rollout_db.src.planner import preflight
from audit_utils import source_inventory
G=guard()
ARMS=['nominal only','safety baseline','F10','F13','S300','S500','RHR-fair']
RULE=dict(mode='progress',steps=40,speed=.25)
def new_exp(phase):return ('experiment',dict(experiment_uid='exp_teacher_student_macflow_v1f_'+phase,name='Closed-loop relabeling and public noise '+phase,path=str(HERE/phase),protocol_hash=sha(HERE/'PREREGISTRATION.md'),code_hash=sha(__file__),metadata_json=canonical(dict(phase=phase,preregistered=True))))
def create(phase):
 assert phase in ['pilot','main'];assert (HERE/'TRAINING_FROZEN.receipt.json').exists()
 if phase=='main':assert (HERE/'CHOICES_FROZEN.receipt.json').exists() and (HERE/'CAPABILITY_FROZEN.receipt.json').exists()
 gate=read(HERE/'DISTILLATION_GATE.json');names=gate['valid_names'] if phase=='pilot' else read(HERE/'CAPABILITY_GATE.json')['eligible'];states=read(HERE/('PILOT_STATES.json' if phase=='pilot' else 'EVAL_STATES.json'));base=read(D/'decentralized_pipeline_v1/main_protocol.json');ts=read(HERE/'TEACHERS.json');registered=[new_exp(phase)];cases=[];protocol=dict(experiment_uid='exp_teacher_student_macflow_v1f_'+phase,phase=phase,states=states,controller_order=names,base_profiles={},eta=base['eta'])
 for name in names:
  rec=next(r for r in gate['controllers'] if r['name']==name);i=rec['index'];variant=rec['variant'];v=dict(base['base_profiles'][ts[i]['name']]['payload']);v.update(flow_checkpoint_path=rec['checkpoint'],flow_checkpoint_sha256=rec['checkpoint_sha256']);protocol['base_profiles'][name]=dict(payload=v)
  for arm in (['safety baseline','S500'] if phase=='pilot' else ARMS):
   j=13 if arm in ['F13','S300','S500'] else 10 if arm=='F10' else None;ea=base['eta'][j] if j is not None else [0.,0.,0.];eb=base['eta'][10] if arm in ['S300','S500'] else ea;switch=int(arm[1:]) if arm in ['S300','S500'] else 700
   payload=dict(v,family='round30f_student',variant=variant,index=i,arm=arm,eta_schedule=dict(a=ea,b=eb,switch_step=switch),rule=RULE if arm=='RHR-fair' else None,pipeline_sha256=sha(HERE/'pipeline.py'),student_source_sha256=sha(HERE/'student.py'),execution_source_sha256=sha(HERE/'evaluate.py'),fair_rhr_sha256=sha(HERE/'fair_rhr.py') if arm=='RHR-fair' else None,noise_tag=3006 if variant=='V3' else None,rng='key_for(state,seed,step); public fold_in(key,3006) or private fold_in(key,i)',jax_enable_x64=False)
   ctl=uid('ctl',payload);eid,values,hx=eta_identity(ea);registered.extend([('controller_config',dict(controller_uid=ctl,scenario_uid=v['scenario_uid'],flow_checkpoint_sha256=rec['checkpoint_sha256'],orthoflow3_sha256=v['orthoflow3_sha256'],safety_config_hash=uid('safe',dict(source=sha(HERE/'pipeline.py'),radius=1.5,nominal_only=arm=='nominal only')),horizon=str(v['horizon']),dt=str(v['dt']),success_semantics_version=v['success_semantics'],conditioning_version='own_local_obs_id_joint_noise' if variant=='V3' else 'own_local_obs_id_private_noise',rng_semantics_version=payload['rng'],config_json=canonical(payload),compatibility_quality='EXACT_PROFILE')),('eta',dict(eta_uid=eid,eta1=values[0],eta2=values[1],eta3=values[2],canonical_hex=hx))])
   for si,s in enumerate(states):
    for seed in ([0] if phase=='pilot' else [0,1]):
     sk=canonical(dict(future_index=seed));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=ctl,seed=sk));cases.append(dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=ctl,seed_key=sk,base_controller_name=name,teacher_name=ts[i]['name'],student_index=i,variant=variant,checkpoint=rec['checkpoint'],checkpoint_sha256=rec['checkpoint_sha256'],arm=arm,state_index=si,seed=seed,eta_a=ea,eta_b=eb,switch_step=switch))
 write(HERE/f'{phase}_protocol.json',protocol);write(HERE/f'{phase}_CASES.json',cases);register(phase,registered)
 manifest=HERE/f'{phase}_PLANNED_ROLLOUTS.json';write(manifest,dict(requests=[{k:c[k] for k in ['state_uid','eta_uid','controller_uid','seed_key']} for c in cases]));pf=preflight(manifest);assert pf['summary']['ambiguous']==0;write(HERE/f'{phase}_CACHE_PREFLIGHT.json',pf);missing=[];reuse=[]
 with connect(True) as db:
  for c in cases:
   row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
   if row:assert not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE';reuse.append(dict(case=c,row=dict(row)))
   else:missing.append(c)
 write(HERE/f'{phase}_MISSING_CASES.json',missing);write(HERE/f'{phase}_REUSE_CASES.json',reuse);seal(f'{phase}_PREFLIGHT.receipt.json',[manifest,HERE/f'{phase}_CACHE_PREFLIGHT.json',HERE/f'{phase}_MISSING_CASES.json',HERE/f'{phase}_protocol.json',HERE/f'{phase}_CASES.json'],before_submission=True,missing=len(missing),reuse=len(reuse));source_inventory(f'imports_design_{phase}.json');print(phase,len(missing),flush=True)
if __name__=='__main__':create(sys.argv[1])
