"""A fixed8-state sparse diagnostic, only after confirmation has been sealed."""
from common import *
from candidates import draw
from frozen_networks import stable_int
from design import action_case,freeze

def plan():
    assert not (HERE/'POSTHOC_DRAW.receipt.json').exists();revision=read(HERE/'PROMPT_REVISION_ACTIVE.json');states=read(HERE/'NEW_EVAL_STATES.json');rows=[]
    for n in read(HERE/'OPENING_CONFIG.json')['unseen_order']:
        seed=stable_int('crosstype3-posthoc-states',revision['diagnostic_states_seed'],n);selected=sorted(np.random.default_rng(seed).choice(48,8,replace=False).tolist())
        rows.append(dict(controller=n,state_draw_seed=seed,state_indices=selected,states=[dict(state_index=i,state_uid=states[i]['state_uid'],candidate_indices=draw(n,states[i]['state_uid'])) for i in selected]))
    write(HERE/'POSTHOC_DRAW.json',dict(utc=utc(),controllers=rows,master_seed=revision['diagnostic_states_seed'],seed_keys=[0],only_after_confirmatory_receipt=True))
    seal('POSTHOC_DRAW.receipt.json',[HERE/'POSTHOC_DRAW.json',HERE/'posthoc_design.py',HERE/'candidates.py',HERE/'PROMPT_REVISION_ACTIVE.receipt.json'],before_any_new_evaluation_outcome_read=True)

def main():
    for path,h in read(HERE/'CONFIRMATORY_RESULTS.receipt.json')['files'].items():assert sha(path)==h,path
    assert read(HERE/'CONFIRMATORY_RESULTS.json')['diagnostic_labels_read'] is False
    base=read(HERE/'EVALUATION_PROTOCOL_BASE.json');eligible=set(read(HERE/'UNSEEN_CAPABILITY.json')['eligible']);states=read(HERE/'NEW_EVAL_STATES.json');candidates=read(HERE/'CANDIDATES.json')['states'][48:];rows=[];profiles={}
    for r in read(HERE/'POSTHOC_DRAW.json')['controllers']:
        name=r['controller']
        if name not in eligible:continue
        for local,record in enumerate(r['states']):
            si=record['state_index'];s=states[si];assert record['state_uid']==s['state_uid']
            for slot,ci in enumerate(record['candidate_indices']):
                c,cp=action_case(base,name,s,candidates[si]['actions'][ci],0,si,ci);c.update(diagnostic_state_index=local,training_slot=slot);rows.append(c);profiles[cp['controller_uid']]=cp
    base['controller_order']=[n for n in base['controller_order'] if n in eligible]
    freeze('diagnostic',dict(base,states=states,phase='diagnostic'),rows,profiles)
    seal('POSTHOC_LABEL_DESIGN.receipt.json',[HERE/'diagnostic_PREFLIGHT.receipt.json',HERE/'diagnostic_EXACT_BUDGET.json',HERE/'CONFIRMATORY_RESULTS.receipt.json',HERE/'POSTHOC_DRAW.receipt.json',HERE/'posthoc_design.py'],after_confirmation_sealed=True,development_only=True)

if __name__=='__main__':plan() if sys.argv[1]=='plan' else main()
