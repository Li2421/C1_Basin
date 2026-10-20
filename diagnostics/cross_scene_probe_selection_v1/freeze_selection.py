"""Open pilot labels only; freeze probe features and choices before main labels."""
from core import *
from collections import defaultdict

def main():
    prereg();gate();p=read(HERE/'protocol.json');pilot=read(HERE/'pilot_CASES.json');probes=read(HERE/'probe_CASES.json');assert not (HERE/'main/raw').exists() or not list((HERE/'main/raw').glob('*.json'))
    results=defaultdict(lambda:defaultdict(list))
    for c in pilot:
        r=read(HERE/'pilot/raw'/f"{c['rollout_uid']}.json");assert r['case']==c and r['success'] in (0,1);results[c['variant']][c['arm']].append(r)
    validity={};active=[]
    for variant in sorted(read(HERE/'GEOMETRIES.json')['scenarios']):
        if variant.endswith('_original'):continue
        arm_results={a:results[variant][a] for a in ('F10','F15')};assert all(len(v)==16 for v in arm_results.values())
        rates={a:sum(r['success'] for r in v)/16 for a,v in arm_results.items()};unknown={a:sum(r['numerical_failure'] for r in v) for a,v in arm_results.items()};eligible=not(rates['F10']<.20 and rates['F15']<.20)
        validity[variant]=dict(rates=rates,unknown=unknown,eligible=eligible,criterion='exclude only when both observed successes/16 < 0.20')
        if eligible:active.append(variant)
    write(HERE/'VALIDITY_GATE.json',dict(utc=utc(),active=active,variants=validity,source_cases=len(pilot),geometry_pass=all(q['valid'] for q in read(HERE/'GEOMETRIES.json')['checks'])))
    by=defaultdict(lambda:defaultdict(list))
    for c in probes:
        r=read(HERE/'probe/raw'/f"{c['rollout_uid']}.json");assert r['case']==c and not r['contains_success_label'] and not r['contains_termination_label']
        if c['variant'] in active or c['variant'].endswith('_original'):
            assert r['error'] is None and r['steps']==c['probe_steps'],'Probe incomplete: stop before evaluation labels'
            by[c['variant']][c['eta_index']].append(r)
    choice={};features={}
    for variant,v in sorted(by.items()):
        assert len(v[13])==16 and len(v[15])==16
        f={a:dict(stall=float(np.mean([r['stall_fraction'] for r in v[a]])),progress=float(np.mean([r['progress_fraction'] for r in v[a]])),minimum_pair_distance=float(np.mean([r['minimum_pair_distance'] for r in v[a]]))) for a in (13,15)}
        selected='S13_t1' if f[15]['stall']>f[13]['stall']+.05 else 'S15_t2';choice[variant]=selected;features[variant]=f
    freeze(HERE/'PROBE_SELECTION_FREEZE.json',dict(utc=utc(),rule='S13_t1 iff stall15 > stall13 + 0.05, else S15_t2',active_variants=active,features=features,selections=choice,probe_source_count=sum(len(v[a]) for v in by.values() for a in (13,15)),pilot_gate_sha256=sha(HERE/'VALIDITY_GATE.json'),probe_preflight_receipt_sha256=sha(HERE/'probe_PREFLIGHT.receipt.json'),probe_official_preflight_sha256=sha(HERE/'probe_OFFICIAL_PREFLIGHT.json'),probe_code_sha256=sha(HERE/'probe.py'),run_code_sha256=sha(HERE/'run.py'),before_new_evaluation_labels=True))
    write(HERE/'PROBE_SELECTION_FREEZE.receipt.json',dict(utc=utc(),sha256=sha(HERE/'PROBE_SELECTION_FREEZE.json'),prereg_sha256=sha(HERE/'PREREGISTRATION.md'),before_new_evaluation_labels=True))
    print('Frozen',len(active),'active variants',choice)
if __name__=='__main__':main()
