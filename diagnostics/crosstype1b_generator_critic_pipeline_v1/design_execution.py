"""Seal generated-action requests, excluding untouched new states until freeze."""
from common import *
from registry import register
from shared_rollout_db.src.rollout_db import uid,canonical,eta_identity
import copy


DEPENDENCIES=('generated_execution.py','rollout_runtime.py','zoo_runtime.py','zoo_models.py','ippo.py','controllers.py',
    'historical_controllers.py','legacy_controllers.py','learned.py','gaussian.py','right_hand.py','trigger.py','progress.py','core.py','common.py','worker.py')


def design(stage,split,which):
    assert not (HERE/f'{stage}_EXECUTION_FROZEN.receipt.json').exists()
    for path,digest in read(HERE/'CANDIDATES.receipt.json')['files'].items():assert sha(path)==digest
    base=read(HERE/'BASE_PROTOCOL.json');candidate=read(HERE/'CANDIDATES.json')
    names=base['controller_order']
    if split=='training':
        assert which in ('pilot','remaining','seed1_extension')
        if which=='pilot':names=[n for n in names if base['types'][n] in ('Flow','diffusion','ring_ippo')]
        if which=='remaining':
            assert (HERE/'PILOT_REPORT.receipt.json').exists()
            names=[n for n in names if base['types'][n] not in ('Flow','diffusion','ring_ippo')]
        if which=='seed1_extension':assert (HERE/'TRAINING_SEED1_BUDGET_AUTHORIZATION.json').exists()
        states=read(HERE/'TRAINING_STATES.json');selected=range(10);seeds=(1,) if which=='seed1_extension' else (0,)
        if which=='seed1_extension':states=states[16:]
    else:
        assert split=='new_evaluation' and which=='all'
        for receipt in ('FROZEN_PIPELINE.receipt.json','NEW_EVAL_CHOICES.receipt.json'):
            for path,digest in read(HERE/receipt)['files'].items():assert sha(path)==digest
        states=read(HERE/'NEW_EVAL_STATES.json');selected=range(14);seeds=(0,1)
    p={**base,'states':states,'controller_order':names,'phase':split};rows=[];profiles={}
    by={r['state_uid']:r for r in candidate['states']}
    for name in names:
        for si,s in enumerate(states):
            for aidx in selected:
                a=by[s['state_uid']]['actions'][aidx] if aidx<13 else dict(name='nominal_only',eta_a=[0.,0.,0.],eta_b=[0.,0.,0.],switch_mode='fixed',safety_baseline=False)
                v=copy.deepcopy(base['profiles'][name]['payload'])
                if aidx==13:v.update(conditioning='nominal_only_without_projection_or_correction',execution_semantics='round38_nominal_only_v1')
                elif a['switch_mode']=='progress':
                    v['generator_progress_schedule']=dict(early_eta='request_eta_uid',late_eta=a['eta_b'],pstar=.9681620884830543,fallback=600,
                        progress_implementation_sha256=sha(HERE/'progress.py'),wrapper_sha256=sha(HERE/'generated_execution.py'))
                elif a['switch_mode']=='step500':v['eta_schedule']=dict(eta_a=a['eta_a'],eta_b=a['eta_b'],switch_step=500,action_index_rule='a if index<t else b')
                cid=uid('ctl',v);profiles[cid]=dict(controller_uid=cid,payload=v)
                eid=eta_identity(a['eta_a'])[0]
                for seed in seeds:
                    sk=canonical(dict(future_index=seed));rid=uid('roll',dict(state=s['state_uid'],eta=eid,controller=cid,seed=sk))
                    rows.append(dict(rollout_uid=rid,state_uid=s['state_uid'],eta_uid=eid,controller_uid=cid,seed_key=sk,
                        base_controller_name=name,family=base['types'][name],arm=a['name'],candidate_index=aidx,seed=seed,
                        eta_a=a['eta_a'],eta_b=a['eta_b'],switch_mode=a['switch_mode'],switch_step=500 if a['switch_mode']=='step500' else 700,
                        state_index=by[s['state_uid']]['state_index'],phase_state_index=si,placement='FF' if base['types'][name]=='Flow' else 'TT',rhr_rule=None))
    register(stage,p,rows,profiles)
    seal(f'{stage}_EXECUTION_FROZEN.receipt.json',[HERE/f'{stage}_protocol.json',HERE/f'{stage}_CASES.json',HERE/f'{stage}_PREFLIGHT.receipt.json',
         HERE/'CANDIDATES.receipt.json',*[HERE/name for name in DEPENDENCIES]],split=split,selection=which,before_rollouts=True)


if __name__=='__main__':design(*sys.argv[1:])
