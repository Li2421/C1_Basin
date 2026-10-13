"""Four-chain B15 candidate interaction using frozen Q16 labels only."""
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'part2_closed_loop_sensitivity_validation_20261012'
CHAINS=('TT','TF','FT','FF')
PAIRS=(('TT','TF'),('FT','FF'),('TT','FT'),('TF','FF'),('TT','FF'))

def main():
    with (PREV/'b15_by_candidate.csv').open(newline='') as f:source=list(csv.DictReader(f))
    table=defaultdict(dict)
    for r in source:table[(r['scene'],r['uid'],r['eta_index'])][r['placement']]=r['status']
    rows=[];stats=Counter()
    for (scene,uid,ei),a in table.items():
        row=dict(scene=scene,uid=uid,eta_index=int(ei),**{k:a[k] for k in CHAINS})
        for old,new in PAIRS:
            outcome='unknown' if 'UNKNOWN' in (a[old],a[new]) else 'gain' if (a[old],a[new])==('FALSE','TRUE') else 'loss' if (a[old],a[new])==('TRUE','FALSE') else 'same'
            row[f'{old}_{new}']=outcome;stats[f'{scene}:{old}:{new}:{outcome}']+=1
        row['binary_interaction']='' if 'UNKNOWN' in a.values() else int(a['FF']=='TRUE')-int(a['FT']=='TRUE')-int(a['TF']=='TRUE')+int(a['TT']=='TRUE')
        rows.append(row)
    with (HERE/'four_chain_interaction.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(pair_counts=dict(stats),interaction_counts={})
    for scene in ('toy_give_way','ring_exchange'):
        v=[int(r['binary_interaction']) for r in rows if r['scene']==scene and r['binary_interaction']!='']
        summary['interaction_counts'][scene]=dict(n_complete=len(v),histogram=dict(Counter(v)),mean=sum(v)/len(v))
    (HERE/'four_chain_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(summary['interaction_counts'])

if __name__=='__main__':main()
