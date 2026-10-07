"""Independent CSV aggregation and reproducibility checks, no task execution."""
import collections
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())


def main():
    groups=collections.defaultdict(dict);row_count=0
    with gzip.open(HERE/'seed_outcomes.csv.gz','rt',newline='') as f:
        for row in csv.DictReader(f):
            row_count+=1
            k=(row['cohort'],row['state_uid'],int(row['eta_index']),row['chain'])
            seed=int(row['seed']);assert seed not in groups[k]
            groups[k][seed]=row
    counts={}
    for key,rows in groups.items():
        assert sorted(rows)==list(range(16))
        s=sum(int(r['success']) for r in rows.values());u=sum(int(r['numerical']) for r in rows.values())
        counts[key]=1 if s>=15 else 0 if s+u<15 else -1
    table=collections.Counter();state_d=collections.Counter();state_change=collections.Counter()
    for (cohort,state,eta,ch),rows in groups.items():
        if cohort!='primary' or ch!='TT':continue
        other=groups[cohort,state,eta,'FF']
        for seed in range(16):
            assert rows[seed]['seed_key']==other[seed]['seed_key']
            assert rows[seed]['controller_uid']!=other[seed]['controller_uid']
            for name in ('eta1','eta2','eta3'):assert rows[seed][name]==other[seed][name]
        a=counts[cohort,state,eta,'TT'];b=counts[cohort,state,eta,'FF'];assert a>=0 and b>=0
        table[a,b]+=1;state_d[state]+=b-a;state_change[state]+=a!=b
    a=next(r for r in read(HERE/'summary.json') if (r['cohort'],r['scene'],r['contrast'])==('primary','all','TT->FF'))
    assert a['both_failure']==table[0,0] and a['rescue']==table[0,1]
    assert a['break_count']==table[1,0] and a['both_success']==table[1,1]
    assert a['states_changed']==sum(x>0 for x in state_change.values())
    assert a['volume_increase_states']==sum(x>0 for x in state_d.values())
    assert a['volume_decrease_states']==sum(x<0 for x in state_d.values())
    assert a['volume_tie_states']==sum(x==0 for x in state_d.values())
    assert a['volume_delta']==sum(state_d.values())/512
    for row in csv.DictReader((HERE/'cell_membership.csv').open()):
        key=(row['cohort'],row['state_uid'],int(row['eta_index']),row['chain'])
        assert counts[key]==int(row['B15'])
    reps=read(HERE/'figure_selection.json')['states'];states=read(HERE/'inputs/primary_states.json')
    for scene,uid in reps.items():
        expected=min([s for s in states if s['scenario']==scene],key=lambda s:hashlib.sha256(('basin-map-v1|'+s['uid']).encode()).hexdigest())['uid']
        assert uid==expected
    for name,hx in read(HERE/'snapshot_integrity.json').items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==hx
    assert row_count==read(HERE/'cache_audit.json')['total_archived_records']==33792
    doc=dict(status='PASS',independent_csv_contingency={str(k):v for k,v in table.items()},
             raw_rows=row_count,all_seed_sets_complete=True,primary_pairing_exact=True,
             membership_csv_matches_independent_seed_aggregation=True,summary_matches_independent_aggregation=True,
             figure_selection_matches_outcome_blind_hash=True,snapshot_integrity=True,
             new_rollouts=0,python=platform.python_version(),
             packages={p:importlib.metadata.version(p) for p in ('numpy','scipy')},
             unit_tests='test_analysis.py: 4 checks for unknown outcomes, state permutation, contingency identity, multiplicity correction')
    (HERE/'verification.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps(doc))


if __name__=='__main__':main()
