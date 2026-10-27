"""Compare cache reuse against independently computed saved twin diagnoses."""
from experiment import *
from diagnosis_alias_cache import reuse,payload
from copy import deepcopy

def main():
    cases=read(HERE/'central_baseline_lit_crowded_random_CASES.json');seen=set();checks=[];rejection_checks=[]
    for c in cases:
        if c['seed']!=1:continue
        path=HERE/'central_baseline_lit_crowded_random'/'raw'/f"{c['rollout_uid']}.json"
        raw=read(path)
        if not raw.get('deterministic_source'):continue
        key=(c['controller_payload']['nominal']['type_code'],c['state']['N'],raw['outcome'])
        if key in seen:continue
        seen.add(key);actual=reuse(raw);assert actual is not None
        expected=read(HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json")
        match=canonical(actual['result'])==canonical(payload(expected));assert match
        checks.append(dict(rollout_uid=c['rollout_uid'],case_group=key,passed=match,source=actual['provenance']))
        if not rejection_checks:
            for changed in ('trace_sha256','responsibility','case'):
                negative=deepcopy(raw)
                if changed=='trace_sha256':negative['trace_sha256']='invalid trace'
                elif changed=='responsibility':negative['responsibility']=1.
                else:negative['case']['eta_a']=[999.,0.,0.]
                assert reuse(negative) is None
                rejection_checks.append(changed)
    assert len(checks)>=21,len(checks)
    write(HERE/'DIAGNOSIS_ALIAS_CACHE_TESTS.json',dict(utc=utc(),passed=True,checks=checks,rejected_changed_semantics=rejection_checks,independent_existing_diagnoses_compared=len(checks),new_physical_steps=0,new_research_rollouts=0,new_training=0))
    print('Independent saved twin diagnosis checks',len(checks),'rejected semantic mismatches',rejection_checks,flush=True)

if __name__=='__main__':main()
