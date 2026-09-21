"""Frozen-score component deletions on the same 64-state candidate pool."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/c1_frozen_unseen_64'
OUT=ROOT/'results/c1_pretraining_audits';OUT.mkdir(exist_ok=True)
SCORES={'P':lambda c:c['P'],'P+S':lambda c:c['P']+c['Stwo'],
    'P+g':lambda c:c['P']+c['g'],'P+g+S':lambda c:c['score'],
    'literal_P+g+S':lambda c:c['P']+c['g']+c['Stwo']}

def main():
    protocol=json.loads((BASE/'protocol.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protocol['hashes'].items())
    states=[]
    for rid in protocol['ids']:
        rows=[json.loads(f.read_text()) for f in (BASE/'traces'/str(rid)).glob('*.json')]
        good=[r for r in rows if r['costs'] is not None];assert len(rows)==161
        selected={name:min(good,key=lambda r:(score(r['costs']),r['cid'])) for name,score in SCORES.items()}
        states.append(dict(rid=rid,coverage=any(r['outcome']['success'] for r in rows),selected=selected))
    summary={}
    for name in SCORES:
        rr=[s['selected'][name] for s in states];success=sum(r['outcome']['success'] for r in rr)
        summary[name]=dict(success_selection=success,coverage_denominator=sum(s['coverage'] for s in states),
            final_success=success,deadlock=sum(not r['outcome']['success'] and r['outcome']['deadlock'] for r in rr),
            timeout=sum(not r['outcome']['success'] and not r['outcome']['deadlock'] for r in rr),
            mean_stagnation_seconds=float(np.mean([r['outcome']['stagnation_seconds'] for r in rr])),
            total_stagnation_seconds=float(sum(r['outcome']['stagnation_seconds'] for r in rr)))
    changes={}
    for before,after in [('P','P+g'),('P','P+S'),('P+S','P+g+S'),('P+g','P+g+S'),('P+g+S','literal_P+g+S')]:
        cases=[]
        for s in states:
            a,b=s['selected'][before],s['selected'][after]
            if a['cid']==b['cid']:continue
            cases.append(dict(rid=s['rid'],before_cid=a['cid'],after_cid=b['cid'],before_costs=a['costs'],after_costs=b['costs'],
                before_outcome=a['outcome'],after_outcome=b['outcome'],
                stagnation_change=b['outcome']['stagnation_seconds']-a['outcome']['stagnation_seconds']))
        changes[before+' -> '+after]=dict(changed=len(cases),rescued=sum(not c['before_outcome']['success'] and c['after_outcome']['success'] for c in cases),
            harmed=sum(c['before_outcome']['success'] and not c['after_outcome']['success'] for c in cases),
            total_stagnation_change=sum(c['stagnation_change'] for c in cases),cases=cases)
    result=dict(definitions={'S':'Stwo=U*S_raw*Ctwo; original frozen fusion full=P+mean(g+Stwo-g*Stwo)',
        'literal':'Additional arithmetic diagnostic only; not substituted for frozen full scorer.'},summary=summary,changes=changes,states=states)
    (OUT/'ablation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,changes={k:{a:b for a,b in v.items() if a!='cases'} for k,v in changes.items()}),indent=2))

if __name__=='__main__':main()
