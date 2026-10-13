"""Four frozen-anchor factorial policies. No source-tree edits or relaxed safety."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from poc.field_controller import ROOT,sha,digest

BASE=ROOT/'validation_stage2';V2=ROOT/'validation_v2'
CHAINS=('TT','FT','TF','FF')
def descriptor(rt,chain):
    if chain not in CHAINS:raise ValueError(chain)
    safety_source=ROOT/'source/single_integrator/cbf.py' if rt.scenario=='toy_give_way' else ROOT/'source/shared_control/hard_projection.py'
    return {'schema':'stage2_factorial_controller_v1','scenario':rt.scenario,'chain':chain,
      'checkpoint_sha256':sha(rt.checkpoint),'basis_sha256':sha(ROOT/'source/shared_control/basis_families.py'),
      'safety_source_sha256':sha(safety_source),'field_base_sha256':sha(ROOT/'poc/field_controller.py'),
      'scenario_adapter_sha256':sha(V2/'adapters.py'),
      'factorial_controller_sha256':sha(BASE/'factorial_controller.py'),'protocol_sha256':sha(BASE/'protocol.json'),
      'cbf':rt.cbf.to_dict(),'pseudo_steps':rt.steps,'future_root':2026100402,
      'anchor':'same native Flow noise and original a_ref for all cells; constant physical delta',
      'eta_schedule':'fixed across physical episode, basis recomputed per physical step',
      'terminal_closure_for_FT':True}
def uid(rt,chain):return 'stage2_ctl_'+digest(descriptor(rt,chain))
def action(rt,env,key,eta,chain):
    if chain=='TT':
        a,cert,calls=rt.action(env,key,eta,'old')
        return a,cert,calls,{'pre_terminal_violation':None}
    if chain=='FF':
        a,cert,calls=rt.action(env,key,eta,'field')
        return a,cert,calls,{'pre_terminal_violation':None}
    obs,safety,flow,safe,delta,_=rt.prepare(env,key,eta)
    if chain=='FT':
        base,_=rt.integrate(env,obs,key,np.zeros_like(delta),safety,project=True)
        target=base+delta
    elif chain=='TF':
        target,_=rt.integrate(env,obs,key,delta,safety,project=False)
    else:raise ValueError(chain)
    precert=safety.certificate(target)
    a,_=safety.project(target)
    cert=safety.assert_safe(a)
    return a,cert,safety.calls,{'pre_terminal_violation':precert['max_violation']}
