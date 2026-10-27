"""Saved-trace doorway component directions near nominally stopped goal approaches."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from result_inventory import find_raw
from states import LiteratureConfig
from distributed_safety import local_basis
from correction_revision_20261021 import distributed_fields,central_fields,contact_radius
from shared_control.basis_families import get_basis_family
from shared_control.hard_projection import HardProjectionConfig
import numpy as np,collections

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']});verify_receipt('INSTANCE_CLASSES_REVISED.receipt.json');verify_receipt('CORRECTION_REVISION.receipt.json')
 classes=read(HERE/'INSTANCE_CLASSES_REVISED.json')['instances'];eta=np.asarray(read(HERE/'FIXED_ETA.json')['10']['values']);basis=get_basis_family('orthoflow3');hp=HardProjectionConfig();examples=[];sources=[]
 for layer in ('central','distributed'):
  failed={r['controller'] for r in classes if (r['scene'],r['layer'])==('lit_doorway',layer) and r['remove_in_revised_exclusion']}
  baseline=read(HERE/f'{layer}_baseline_lit_doorway_CASES.json');selected=collections.Counter();fixed={}
  for sem,prefix in [('original',''),('revised','revised_')]:
   fixed[sem]={(c['controller_payload']['nominal']['controller'],c['state']['index'],c['seed']):c for c in read(HERE/f'{layer}_{prefix}fixed_lit_doorway_CASES.json') if c['method']=='fixed_eta10'}
  for c in baseline:
   ctl=c['controller_payload']['nominal']['controller']
   if ctl not in failed or selected[ctl]>=2:continue
   path=find_raw(c['rollout_uid'],scene='lit_doorway');raw=read(path)
   if raw['success'] or raw['numerical_failure']:continue
   cfg=LiteratureConfig(**c['scenario_payload']['config'])
   with np.load(raw['trace_path']) as z:
    dist0=np.linalg.norm(z['positions'][-1]-z['goals'],axis=-1);tail=slice(max(0,len(z['nominal'])-50),None);baseline_info=dict(final_distances=dist0.tolist(),last50_nominal_max=float(np.linalg.norm(z['nominal'][tail],axis=-1).max(initial=0)),last50_first_projection_change_max=float(np.linalg.norm(z['safe'][tail]-z['nominal'][tail],axis=-1).max(initial=0)))
   example=dict(controller=ctl,layer=layer,state=c['state']['index'],seed=c['seed'],baseline=baseline_info,semantics={});sources.append(dict(raw=str(path),raw_sha256=sha(path),trace_sha256=raw['trace_sha256']))
   for sem in ('original','revised'):
    fc=fixed[sem][(ctl,c['state']['index'],c['seed'])];fp=find_raw(fc['rollout_uid'],scene='lit_doorway');fr=read(fp);summary={label:dict(agent_steps=0,weighted_radial_component_sums=np.zeros(3),weighted_component_norm_sums=np.zeros(3),actual_goalward_velocity_sum=0.,first_safe_goalward_velocity_sum=0.,relative_zero_agent_steps=0,positive_actual_goalward_steps=0) for label in ('near_goal','near_goal_nominal_silent_and_first_projection_unchanged')}
    with np.load(fr['trace_path']) as z:
     for t,v in enumerate(z['safe']):
      p=z['positions'][t];g=z['goals'];err=g-p;dist=np.linalg.norm(err,axis=-1);direction=err/np.maximum(dist[:,None],1e-30)
      if layer=='central':fields=np.asarray((central_fields(p,g,v,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg),basis) if sem=='revised' else basis.compute(p,g,v,cfg.max_speed)).values)
      else:fields=np.asarray([distributed_fields(p,g,v,i,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg)) if sem=='revised' else local_basis(p,g,v,i,np.inf,cfg.max_speed) for i in range(len(p))]).transpose(1,0,2)
      weighted=eta[:,None,None]*fields;radial=np.einsum('kij,ij->ki',weighted,direction);norms=np.linalg.norm(weighted,axis=-1);near=(dist>cfg.goal_tolerance)&(dist<=np.maximum(3*cfg.goal_tolerance,dist0));silent=(np.linalg.norm(z['nominal'][t],axis=-1)<=1e-6)&(np.linalg.norm(v-z['nominal'][t],axis=-1)<=hp.intervention_tol)
      for label,mask in [('near_goal',near),('near_goal_nominal_silent_and_first_projection_unchanged',near&silent)]:
       s=summary[label];s['agent_steps']+=int(mask.sum());s['weighted_radial_component_sums']+=radial[:,mask].sum(-1);s['weighted_component_norm_sums']+=norms[:,mask].sum(-1);actual=np.sum(z['actions'][t]*direction,axis=-1);safe_radial=np.sum(v*direction,axis=-1);s['actual_goalward_velocity_sum']+=float(actual[mask].sum());s['first_safe_goalward_velocity_sum']+=float(safe_radial[mask].sum());s['relative_zero_agent_steps']+=int(np.sum(mask&(np.linalg.norm(fields[2],axis=-1)==0)));s['positive_actual_goalward_steps']+=int(np.sum(mask&(actual>1e-9)))
    example['semantics'][sem]=dict(success=fr['success'],unknown=fr['numerical_failure'],frames=summary);sources.append(dict(raw=str(fp),raw_sha256=sha(fp),trace_sha256=fr['trace_sha256']))
   selected[ctl]+=1;examples.append(example)
  assert all(selected[c]==2 for c in failed),(layer,selected)
 result=dict(utc=utc(),scope='Two finite failed baseline requests per controller-level-failure instance and layer,paired saved fixed eta10 traces',selection='First baseline cases in sealed manifest order; no choice based on corrected outcome',near_goal_window='Unarrived and distance<=max(3*original_tolerance,paired_baseline_terminal_distance)',silent_nominal_threshold=1e-6,unchanged_first_projection_threshold=hp.intervention_tol,components=['Bgoal','Bperp','Brel'],interpretation='修正推动controller自身停住的实例，不计解开死锁。方向分量在第二次投影之前；实际向目标速度单独报告，不把非线性投影后的因果效应强行按范数分摊。',examples=examples,sources=sources,new_rollouts=0,new_training=0)
 write(HERE/'DOOR_GOAL_PROGRESS_MECHANISM_20261023.json',result)
 text='保存轨迹机制核查：门口controller级失败实例。固定η10在两种语义下与同state/seed的失败基线配对；按封存顺序每实例取2局。近目标窗口及静止阈值在JSON明确列示，零有效帧照实保留。数值为第二次投影前的带权分量沿目标方向的和，实际朝目标动作另外给出；不作分量消融的因果声称。修正推动controller自身停住的实例，不计解开死锁。\n\n|层|controller|state/seed|语义|近目标且名义静止帧|Bgoal方向和|Bperp方向和|Brel方向和|实际朝目标动作和|Brel为零帧|成功|\n|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for ex in examples:
  for sem,entry in ex['semantics'].items():
   s=entry['frames']['near_goal_nominal_silent_and_first_projection_unchanged'];v=s['weighted_radial_component_sums'];text+='| '+' | '.join(map(str,[ex['layer'],ex['controller'],(ex['state'],ex['seed']),sem,s['agent_steps'],*[round(float(x),8) for x in v],round(s['actual_goalward_velocity_sum'],8),s['relative_zero_agent_steps'],entry['success']]))+' |\n'
 (HERE/'DOOR_GOAL_PROGRESS_MECHANISM_20261023.md').write_text(text)
 seal('DOOR_GOAL_PROGRESS_MECHANISM_20261023.receipt.json',[Path(__file__),HERE/'DOOR_GOAL_PROGRESS_MECHANISM_20261023.json',HERE/'DOOR_GOAL_PROGRESS_MECHANISM_20261023.md',HERE/'INSTANCE_CLASSES_REVISED.receipt.json',HERE/'CORRECTION_REVISION.receipt.json'],saved_evidence_only=True,new_rollouts=0,matching_before_outcome_selection=True)
 print('Door goal-progress mechanism examples',len(examples),flush=True)
if __name__=='__main__':main()
