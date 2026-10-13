"""Noise-corrected paired-controller signal under identical current-branch inputs."""
import gzip
import json
import numpy as np
from .study import HERE, read, write


def run():
    d=np.load(HERE/'labels.npz');ix=np.flatnonzero(d['split']=='development')
    outcomes=d['outcome'][ix]
    first=outcomes[...,:8];second=outcomes[...,8:]
    # Retain only fully numeric paired cells; report that restriction explicitly.
    valid=(outcomes>=0).all((1,3))
    q1=(first==1).mean(-1);q2=(second==1).mean(-1)
    delta1=q1[:,1]-q1[:,0];delta2=q2[:,1]-q2[:,0]
    product=delta1*delta2
    # With equiprobable controllers and exactly aliased input, optimal
    # squared-probability prediction incurs delta_true^2 / 4 per paired case.
    state_num=(product*valid).sum(-1)/4
    state_den=valid.sum(-1)
    rng=np.random.default_rng(2026100922);boot=rng.integers(len(ix),size=(20000,len(ix)))
    estimates=state_num[boot].sum(-1)/state_den[boot].sum(-1)
    first_actions={}
    with gzip.open(HERE/'raw_seed_evidence.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);s,c,e,k=row['key']
            if s not in ix:continue
            a=row['result'].get('first_action')
            if a is not None:first_actions[(s,e,k,c)]=np.asarray(a)
    differences=[]
    for (s,e,k,c),value in first_actions.items():
        if c==0 and (s,e,k,1) in first_actions:
            differences.append(float(np.max(np.abs(value-first_actions[(s,e,k,1)]))))
    result=dict(scope='Previously inspected Ring paired-controller DEVELOPMENT states, not new confirmation',
        physical_states=len(ix),complete_numeric_state_eta_cells=int(valid.sum()),
        omitted_numerical_state_eta_cells=int((~valid).sum()),
        first_action_paired_seed_cases=len(differences),first_action_max_absolute_difference=max(differences),
        identical_current_branch='Exact same base Flow function for both full controller programs; this includes all current-branch static queries, not merely one sampled action.',
        independent_half_delta_energy=float((product*valid).sum()/valid.sum()),
        alias_squared_probability_error_lower_bound_estimate=float(state_num.sum()/state_den.sum()),
        alias_squared_probability_error_lower_bound_ci95=np.quantile(estimates,[.025,.975]).tolist(),
        interpretation='An empirical noise-corrected estimate of unavoidable Q error for representations that alias the two controller programs. Does NOT apply to H20 or complete future-branch descriptors.',
        assumptions='Independent seed halves, uniform controller weighting; inference conditional on numeric-complete cells and these two programs.',
        future_branch_feature_difference=read(HERE/'features_audit.json')['future_branch_mean_abs_difference'])
    write(HERE/'information_alias_audit.json',result)
    np.savez_compressed(HERE/'information_alias_bootstrap.npz',indices=boot,lower_bound_estimates=estimates)
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()
