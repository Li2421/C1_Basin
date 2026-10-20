"""Outcome-blind frozen comparisons, sealed before assembled evaluation labels."""
from common import *
from frozen_standard import predict as standard
from frozen_expanded import predict as expanded
from selector import ARMS

def main(cohort):
    target=HERE/f'{cohort.upper()}_COMPARATOR_CHOICES.json';receipt=target.with_suffix('.receipt.json')
    assert not receipt.exists()
    if cohort=='locked':assert (HERE/'FROZEN_PIPELINE.receipt.json').exists()
    with np.load(HERE/f'{cohort.upper()}_FEATURES.npz',allow_pickle=False) as z:
        assert not set(z.files)&{'labels','success','outcome','tuning'}
        names=list(z['names']);r5=z['r5']
    states=read(HERE/'EVAL_STATES.json');choices,scores=standard(states,r5)
    choices['frozen_round38'],scores['frozen_round38']=expanded(states,r5)
    rows={}
    for method,values in choices.items():
        unsupported=[str(n) for n,a in zip(names,values) if str(a) not in ARMS]
        rows[method]=dict(choices={str(n):str(a) for n,a in zip(names,values)},unsupported_current_menu=unsupported,
            available_for_complete_cohort=not unsupported,
            missing_reason=None if not unsupported else 'Original frozen model selected arms outside the preregistered current menu; no substituted choices or invented labels')
    write(target,dict(utc=utc(),cohort=cohort,methods=rows,evaluation_or_tuning_outcomes_read=False))
    seal(receipt.name,[target,HERE/f'{cohort.upper()}_FEATURES.npz',HERE/'seal_comparators.py',HERE/'frozen_standard.py',HERE/'frozen_expanded.py'],
         outcome_blind=True,original_frozen_models_not_retrained=True)
    print(cohort,'frozen comparators sealed',flush=True)

if __name__=='__main__':main(sys.argv[1])
