"""Registered revised execution identities; logical method aliases do not rerun data."""
from experiment import *
from study_cases import make as original_make,qualified
from registry import case,register,verify_receipt
import numpy as np,collections

def exact_vector(a,b):return np.asarray(a,np.float64).tobytes()==np.asarray(b,np.float64).tobytes()

def schedule_method(eta_a,eta_b,theta,index=None):
 fixed=read(HERE/'FIXED_ETA.json')
 if not np.any(eta_a) and not np.any(eta_b):return 'safety'
 for key in ('10','13'):
  if theta is None and exact_vector(eta_a,fixed[key]['values']) and exact_vector(eta_b,fixed[key]['values']):return 'fixed_eta'+key
 if theta==.968 and exact_vector(eta_a,fixed['13']['values']) and exact_vector(eta_b,fixed['10']['values']):return 'human_progress'
 assert index is not None and 1<=index<=11,(index,eta_a,eta_b,theta)
 return f'learning_action_{index:02d}'

def make(s,ctl,layer='distributed',method='safety',seed=0,eta_a=None,eta_b=None,theta=None,menu_index=None,logical_method=None):
 verify_receipt('CORRECTION_REVISION.receipt.json') if not getattr(make,'verified',False) else None
 make.verified=True
 eta_a=[0.,0.,0.] if eta_a is None else eta_a;eta_b=eta_a if eta_b is None else eta_b
 canonical_method=schedule_method(eta_a,eta_b,theta,menu_index)
 base=original_make(s,ctl,layer,canonical_method,seed,eta_a,eta_b,theta)
 zero=canonical_method=='safety'
 if zero:c=base
 else:
  payload={k:v for k,v in base['controller_payload'].items() if k not in ('method','eta_b','theta')}
  payload.update(correction_semantics='contact_conflict_relative_gate_20261021',correction_revision_receipt_sha256=sha(HERE/'CORRECTION_REVISION.receipt.json'),correction_basis='orthoflow3_local_contact_gated' if layer=='distributed' else 'orthoflow3_joint_contact_gated',arrived_flag_broadcast=True)
  c=case(s,base['version'],canonical_method,base['scenario_payload'],payload,seed,eta_a,eta_b,theta,base['state_provenance'])
 c.update(safety_layer=layer,rng_state_uid=base['state_uid'],correction_semantics='eta0_original_execution_equivalent' if zero else 'revised_contact_20261021',canonical_execution_method=canonical_method)
 if menu_index is not None:c['menu_index']=menu_index
 c['method']=logical_method or method
 # Same pattern as frozen learning_design.make_case: reporting method differs
 # from the registered canonical execution schedule. Runtime branches only on
 # nominal_only; all revised arms here are projection+schedule executions.
 assert c['method']!='nominal_only'
 return c

def fixed(layer,scene):
 prereg();verify_receipt('PREREGISTRATION_ADDENDUM_20261021.receipt.json');verify_receipt('CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json')
 fixed=read(HERE/'FIXED_ETA.json');controllers=[c for c in read(HERE/'CONTROLLER_MANIFEST.json') if c['controller'] in qualified(scene)]
 actions=[dict(method='fixed_eta10',eta_a=fixed['10']['values']),dict(method='fixed_eta13',eta_a=fixed['13']['values']),dict(method='human_progress',eta_a=fixed['13']['values'],eta_b=fixed['10']['values'],theta=.968)]
 rows=[make(s,ctl,layer,seed=seed,**a) for ctl in controllers for s in read(HERE/f'FLOW_STATES_{scene}.json') for a in actions for seed in (0,1)]
 expected={'lit_narrow_passage':16896,'lit_crowded_random':29568,'lit_circle_five':4320,'lit_doorway':17280}[scene];assert len(rows)==expected
 assert len({c['rollout_uid'] for c in rows})==len(rows)
 register(f'{layer}_revised_fixed_{scene}',rows)
if __name__=='__main__':fixed(*sys.argv[1:])
