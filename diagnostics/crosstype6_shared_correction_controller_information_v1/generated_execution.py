"""Supply a request's generated early eta to the unchanged execution loop."""
from common import *
from rollout_runtime import execute as historical_execute
import copy


def materialize(c,p):
    current=copy.deepcopy(c);protocol=p
    if c.get('switch_mode')=='progress':
        protocol={**p,'eta':list(p['eta'])}
        protocol['eta'][13]=c['eta_a'];protocol['eta'][10]=c['eta_b']
        current['arm']='Sprog'
    return current,protocol


def execute(c,p,cache,phase):
    current,protocol=materialize(c,p)
    result=historical_execute(current,protocol,cache,phase)
    result['case']=c
    result['generator_action_wrapper_sha256']=sha(__file__)
    return result
