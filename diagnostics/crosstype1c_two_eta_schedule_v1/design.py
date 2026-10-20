"""Exact semantic identities; fixed proposals retain their old cache keys."""
from common import *
from registry import register
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy
DEPENDENCIES=('generated_execution.py','rollout_runtime.py','zoo_runtime.py','zoo_models.py','ippo.py','controllers.py','historical_controllers.py','legacy_controllers.py','learned.py','gaussian.py','right_hand.py','trigger.py','progress.py','core.py','common.py','worker.py','design.py')
def action_case(base,name,s,a,seed,state_index,candidate_index):
    v=copy.deepcopy(base['profiles'][name]['payload']);mode=a['switch_mode']
    if a['name']=='nominal_only':v.update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='round38_nominal_only_v1')
    elif mode=='progress':
        v['generator_progress_schedule']=dict(early_eta='request_eta_uid',late_eta=a['eta_b'],pstar=.9681620884830543,fallback=600,progress_implementation_sha256=sha(HERE/'progress.py'),wrapper_sha256=sha(HERE/'generated_execution.py'))
    elif mode in ('step','step500'):
        v['eta_schedule']=dict(eta_a=a['eta_a'],eta_b=a['eta_b'],switch_step=a['switch_step'],action_index_rule='a if index<t else b')
    cid=uid('ctl',v);eid=eta_identity(a['eta_a'])[0];sk=canonical(dict(future_index=seed));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk))
    c=dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,base_controller_name=name,family=base['types'][name],arm=a['name'],candidate_index=int(candidate_index),seed=seed,eta_a=a['eta_a'],eta_b=a['eta_b'],switch_mode=mode,switch_step=a['switch_step'] if mode in ('step','step500') else 700,state_index=state_index,placement='FF' if base['types'][name]=='Flow' else 'TT',rhr_rule=None)
    return c,dict(controller_uid=cid,payload=v)
def training(stage):
    assert stage in ('pilot','remaining');assert read(HERE/'GENERATED_ACTION_TESTS.json')['passed']
    for path,h in read(HERE/'CANDIDATES.receipt.json')['files'].items():assert sha(path)==h
    if stage=='remaining':assert (HERE/'PILOT_REPORT.receipt.json').exists()
    base=read(HERE/'BASE_PROTOCOL.json');states=read(HERE/'TRAINING_STATES.json');candidates=read(HERE/'CANDIDATES.json')['states'][:48]
    with np.load(HERE/'TRAINING_ACTIONS.npz',allow_pickle=False) as z:indices=z['candidate_indices'];order=z['names']
    pilot={'Flow','diffusion','ring_ippo'};names=[n for n in base['controller_order'] if (base['types'][n] in pilot)==(stage=='pilot')];rows=[];profiles={}
    for name in names:
        i=list(order).index(name)
        for si,s in enumerate(states):
            for local_index,candidate_index in enumerate(indices[i,si]):
                a=candidates[si]['actions'][int(candidate_index)];c,p=action_case(base,name,s,a,0,si,int(candidate_index));c['training_slot']=local_index;c['phase_state_index']=si;rows.append(c);profiles[p['controller_uid']]=p
    protocol={**base,'states':states,'controller_order':names,'phase':'training'}
    register(stage,protocol,rows,profiles)
    seal(f'{stage}_EXECUTION_FROZEN.receipt.json',[HERE/f'{stage}_protocol.json',HERE/f'{stage}_CASES.json',HERE/f'{stage}_PREFLIGHT.receipt.json',HERE/'CANDIDATES.receipt.json',*[HERE/n for n in DEPENDENCIES]],split='training',before_rollouts=True,only_generator_pair_actions=True)
if __name__=='__main__':training(sys.argv[1])
