"""Independent optional queues; all requests are fixed before submission."""
from experiment import *
from study_cases import make,qualified
from registry import case,register,verify_receipt
from optional_cases import converted as original_converted

def converted(base,engine,extra,scenario=None,version=None):
    zero=not any(base['eta_a']) and not any(base['eta_b'])
    if zero:return original_converted(base,engine,extra,scenario,version)
    extra=dict(extra,correction_semantics='contact_conflict_relative_gate_20261021',correction_revision_receipt_sha256=sha(HERE/'CORRECTION_REVISION.receipt.json'),arrived_flag_broadcast=True,revision_factory_sha256=sha(HERE/'optional_cases_correction_revised_20261021.py'))
    payload={k:v for k,v in base['controller_payload'].items() if k not in ('method','eta_b','theta')}
    payload.update(execution_engine=engine,case_factory_sha256=sha(HERE/'optional_cases.py'),rng_identity_rule='Corresponding first-stage physical state UID, generated deterministically by this frozen factory',**extra)
    c=case(base['state'],version or base['version'],base['method'],scenario or base['scenario_payload'],payload,
        base['seed'],base['eta_a'],base['eta_b'],base['theta'],base['state_provenance'])
    c.update(safety_layer='distributed',execution_engine=engine,rng_state_uid=base['state_uid'])
    return c

def build(kind,scene):
    prereg();verify_receipt('MIGRATION_NUMBERS.receipt.json');verify_receipt('CORRECTION_OPTIONAL_EXECUTION_20261021.receipt.json');verify_receipt('CORRECTION_REVISION.receipt.json')
    ctls=read(HERE/'CONTROLLER_MANIFEST.json');states=read(HERE/f'FLOW_STATES_{scene}.json');fixed=read(HERE/'FIXED_ETA.json')
    standard=[dict(method='safety',eta_a=[0.,0.,0.]),dict(method='fixed_eta10',eta_a=fixed['10']['values']),dict(method='fixed_eta13',eta_a=fixed['13']['values']),dict(method='human_progress',eta_a=fixed['13']['values'],eta_b=fixed['10']['values'],theta=.968)]
    if kind in ('dynamics','local_progress'):ctls=[r for r in ctls if r['controller'] in qualified(scene)]
    actions=standard if kind=='dynamics' else standard[-1:] if kind=='local_progress' else [dict(method='nominal_only',eta_a=[0.,0.,0.])] if kind=='nominal' else standard[:2]
    rows=[]
    for ctl in ctls:
        for state in states:
            for a in actions:
                if kind=='full_circle' and a['method']!='safety' and ctl['controller'] not in qualified(scene):continue
                for seed in (0,1):
                    base=make(state,ctl,seed=seed,**a)
                    if kind=='dynamics':
                        verify_receipt('DYNAMICS_IMPLEMENTATION.receipt.json') if not rows else None
                        scenario=dict(base['scenario_payload'],version='paper_dynamics_v2',dynamics_receipt_sha256=sha(HERE/'DYNAMICS_IMPLEMENTATION.receipt.json'))
                        c=converted(base,'paper_dynamics',dict(dynamics_receipt_sha256=sha(HERE/'DYNAMICS_IMPLEMENTATION.receipt.json')),scenario,'paper_dynamics_v2')
                    elif kind=='local_progress':c=converted(base,'local_progress',dict(local_progress_source_sha256=sha(HERE/'runtime_local_progress_correction_revised_20261021.py')))
                    elif kind=='full_circle':
                        assert scene=='lit_circle_five'
                        c=converted(base,'single_integrator',dict(responsibility=1.));c['responsibility']=1.
                    elif kind=='nominal':c=converted(base,'single_integrator',dict(physical_safety_layer='none'))
                    else:raise ValueError(kind)
                    rows.append(c)
    scope=read(HERE/'OPTIONAL_SCOPE.json')
    assert len(rows)==scope['stages'][f'optional_{kind}_{scene}']['planned_requests']
    register(f'optional_{kind}_{scene}',rows)

if __name__=='__main__':build(*sys.argv[1:])
