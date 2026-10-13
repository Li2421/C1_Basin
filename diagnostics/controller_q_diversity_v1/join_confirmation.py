"""Combine already-preflighted candidate and Safety requests into one CPU cap."""
from collections import defaultdict,Counter
from .experiment import HERE,read,write,freeze,sha


def run():
    guard=read(HERE/'confirmation_predictions_frozen.json')
    assert guard['predictions_sha256']==sha(HERE/'confirmation_predictions.npz')
    cases=[];missing=[];counts=Counter()
    for name in ('confirmation_budget','safety_confirmation'):
        folder=HERE/'batches'/name
        pre=read(folder/'exact_identity_preflight.json')
        assert pre['cases_sha256']==sha(folder/'cases.json')
        cases.extend(read(folder/'cases.json'));missing.extend(read(folder/'missing_cases.json'))
        for key in ('requested','exact_reuse','numerical_unknown_not_rerun','genuinely_missing'):counts[key]+=pre.get(key,0)
    key=lambda r:(r['state_index'],r['eta_index'],r['controller_name'],r['seed'],r['scene'])
    cases.sort(key=key);missing.sort(key=key)
    assert len(cases)==len({r['rollout_uid'] for r in cases})==counts['requested']
    groups=defaultdict(list)
    for r in cases:groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    folder=HERE/'batches/confirmation_joint'
    freeze(folder/'cases.json',cases);write(folder/'missing_cases.json',missing)
    freeze(folder/'planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=ss) for (s,e,c),ss in groups.items()]))
    write(folder/'exact_identity_preflight.json',dict(counts,protocol_sha256=sha(HERE/'protocol.json'),cases_sha256=sha(folder/'cases.json')))
    print(dict(counts),flush=True)


if __name__=='__main__':run()
