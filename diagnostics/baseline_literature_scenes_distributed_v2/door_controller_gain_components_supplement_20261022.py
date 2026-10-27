"""Describe saved doorway controller-failure examples; no actors or physics calls."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from result_inventory import find_raw
from distributed_safety import local_basis
from correction_revision_20261021 import distributed_fields,central_fields,contact_radius
from shared_control.basis_families import get_basis_family
from states import LiteratureConfig
import numpy as np,collections

def main():
 verify_receipt('MIGRATION_NUMBERS.receipt.json');classes=read(HERE/'INSTANCE_CLASSES_SUPPLEMENT_20261022.json')['instances'];rows=[];sources=[];fixed=read(HERE/'FIXED_ETA.json')['10']['values'];basis=get_basis_family('orthoflow3')
 for layer in ('central','distributed'):
  eligible={r['controller'] for r in classes if r['scene']=='lit_doorway' and r['layer']==layer and r['remove_in_revised_exclusion']}
  baselines=read(HERE/f'{layer}_baseline_lit_doorway_CASES.json');chosen=collections.Counter()
  index={sem:{(c['controller_payload']['nominal']['controller'],c['state']['N'],c['state']['index'],c['seed']):c for c in read(HERE/f'{layer}_{prefix}fixed_lit_doorway_CASES.json') if c['method']=='fixed_eta10'} for sem,prefix in [('original',''),('revised','revised_')]}
  for c in baselines:
   ctl=c['controller_payload']['nominal']['controller']
   if ctl not in eligible or chosen[ctl]>=2:continue
   path=find_raw(c['rollout_uid'],scene='lit_doorway');raw=read(path)
   if raw['success'] or raw['numerical_failure']:continue
   chosen[ctl]+=1;cfg=LiteratureConfig(**c['scenario_payload']['config']);key=(ctl,c['state']['N'],c['state']['index'],c['seed'])
   with np.load(raw['trace_path']) as z:
    tail=slice(max(0,len(z['nominal'])-50),None)
    example=dict(layer=layer,controller=ctl,state=c['state']['index'],seed=c['seed'],baseline_success=raw['success'],baseline_final_goal_distances=raw['final_goal_distances'],baseline_last50_nominal_max=float(np.linalg.norm(z['nominal'][tail],axis=-1).max(initial=0)),baseline_last50_first_projection_change_max=float(np.linalg.norm(z['safe'][tail]-z['nominal'][tail],axis=-1).max(initial=0)),fixed_results={})
   sources.append(dict(path=str(path),sha256=sha(path),trace_sha256=raw['trace_sha256']))
   for sem in ('original','revised'):
    fc=index[sem][key];fp=find_raw(fc['rollout_uid'],scene='lit_doorway');fr=read(fp);norms=np.zeros(3);steps=0
    with np.load(fr['trace_path']) as z:
     for t,v in enumerate(z['safe']):
      p=z['positions'][t];g=z['goals']
      if layer=='central':fields=(central_fields(p,g,v,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg),basis) if sem=='revised' else basis.compute(p,g,v,cfg.max_speed)).values
      else:
       per=[distributed_fields(p,g,v,i,cfg.max_speed,cfg.goal_tolerance,contact_radius(cfg)) if sem=='revised' else local_basis(p,g,v,i,np.inf,cfg.max_speed) for i in range(len(v))]
       fields=np.asarray(per).transpose(1,0,2)
      norms += [np.linalg.norm(float(weight)*field,axis=-1).sum() for weight,field in zip(fixed,fields)];steps+=1
    example['fixed_results'][sem]=dict(success=fr['success'],timeout=fr['timeout'],unknown=fr['numerical_failure'],steps=steps,sum_weighted_component_norms=dict(zip(('Bgoal','Bperp','Brel'),norms.tolist())),norm_sum_largest_component=('Bgoal','Bperp','Brel')[int(np.argmax(norms))],note='Descriptive pre-second-projection component norm sums; not additive causal attribution after nonlinear projection')
    sources.append(dict(path=str(fp),sha256=sha(fp),trace_sha256=fr['trace_sha256']))
   rows.append(example)
 out=dict(utc=utc(),examples=rows,sources=sources,selection='First two finite failed baseline requests per doorway controller-level-failure instance and layer, paired fixed eta10 saved traces',interpretation='修正推动controller自身停住的实例；不计为解开死锁。分量范数只作轨迹描述，不声称分量消融因果证据',new_rollouts=0,new_training=0)
 write(HERE/'DOOR_CONTROLLER_GAIN_COMPONENTS_20261021.json',out)
 seal('DOOR_CONTROLLER_GAIN_COMPONENTS_20261021.receipt.json',[HERE/'DOOR_CONTROLLER_GAIN_COMPONENTS_20261021.json',Path(__file__)],saved_evidence_only=True)
 print('Saved doorway examples',len(rows),flush=True)
if __name__=='__main__':main()
