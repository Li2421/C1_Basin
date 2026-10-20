from common import *
from zoo_design import profiles,case,freeze,LOCKED
from baselines import LEARNED


def controller_order():
    base=profiles();old=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json');selected=[];counts={}
    for name in old['controller_order']:
        kind=old['types'][name]
        if counts.get(kind,0)>=2:continue
        selected.append(name);counts[kind]=counts.get(kind,0)+1
    assert len(selected)==18 and set(counts.values())=={2}
    for cohort in ('DEVELOPMENT','PPO','LOCKED'):selected.extend(read(HERE/f'{cohort}_CAPABILITY.json')['eligible'])
    assert len(selected)==len(set(selected))
    return selected


def design():
    for receipt in ('FROZEN_PIPELINE.receipt.json','NEW_EVAL_CHOICES.receipt.json'):
        for p,digest in read(HERE/receipt)['files'].items():assert sha(p)==digest
    base=profiles();names=controller_order();states=read(HERE/'NEW_EVAL_STATES.json')
    with np.load(HERE/'NEW_EVAL_CHOICES.npz',allow_pickle=False) as z:assert list(z['names'])==names
    p={**base,'states':states,'controller_order':names,'phase':'new_evaluation'};rows=[];cps={}
    for name in names:
        for si,s in enumerate(states):
            for a in range(17):
                for seed in (0,1):
                    c,cp=case(base,name,s,a,seed);c.update(state_index=48+si,phase_state_index=si)
                    rows.append(c);cps[cp['controller_uid']]=cp
    freeze('new_evaluation',p,rows,cps,split='new_evaluation',full_16_candidates_plus_nominal=True,two_seeds=True,before_confirmation_submission=True)


def few_locked():
    assert (HERE/'new_evaluation_COMPLETE.receipt.json').exists(),'Locked few-label collection only after full confirmatory labels'
    base=profiles();names=read(HERE/'LOCKED_CAPABILITY.json')['eligible'];states=read(HERE/'TRAINING_STATES.json')[:8]
    p={**base,'states':states,'controller_order':names,'phase':'locked_few_labels'};rows=[];cps={}
    for name in names:
        for si,s in enumerate(states):
            for a in range(16):
                c,cp=case(base,name,s,a);c.update(state_index=si,phase_state_index=si);rows.append(c);cps[cp['controller_uid']]=cp
    freeze('locked_few_labels',p,rows,cps,split='locked_few_labels',confirmation_labels_already_complete=True,no_frozen_model_changes=True)


if __name__=='__main__':globals()[sys.argv[1]]()
