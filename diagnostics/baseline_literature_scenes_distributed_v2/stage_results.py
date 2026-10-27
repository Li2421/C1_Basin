"""Stage completeness, the preregistered ability gate and baseline usability."""
from experiment import *
import collections
from result_inventory import logical,qualification

def capability(scene='lit_crowded_random'):
    stage='capability_'+scene;rows,sources=logical(stage)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['case']['controller_payload']['nominal']['controller']].append(r)
    planned=collections.Counter(c['controller_payload']['nominal']['controller'] for c in read(HERE/f'{stage}_CASES.json'))
    out=[]
    for ctl,total in sorted(planned.items()):
        rr=groups[ctl];success=sum(r['success'] for r in rr);unknown=sum(r['numerical_failure'] for r in rr)
        assert len(rr)==total,(ctl,total,len(rr))
        out.append(dict(scene=scene,controller=ctl,planned=total,completed=len(rr),success=success,unknown=unknown,rate_lower=success/total,rate_upper=(success+unknown)/total,qualified=success/total>=.95))
    write(HERE/f'{stage}_QUALIFICATION.json',out);write(HERE/f'{stage}_RESULT_SOURCES.json',sources)
    seal(f'{stage}_QUALIFICATION.receipt.json',[HERE/f'{stage}_QUALIFICATION.json',HERE/f'{stage}_RESULT_SOURCES.json',HERE/'stage_results.py'],no_unknown_retries=True)
    print(scene,'qualified',sum(r['qualified'] for r in out),'of',len(out),flush=True)

def baseline(layer,scene):
    stage=f'{layer}_baseline_{scene}';rows,sources=logical(stage)
    q={r['controller']:r for r in qualification(scene)};groups=collections.defaultdict(list)
    for r in rows:groups[r['case']['controller_payload']['nominal']['controller']].append(r)
    rates={name:dict(lower=sum(r['success'] for r in rr)/len(rr),upper=sum(r['success']+r['numerical_failure'] for r in rr)/len(rr),episodes=len(rr)) for name,rr in groups.items()}
    eligible=[name for name in rates if q.get(name,{}).get('qualified',False)]
    informative=bool(eligible) and any(rates[name]['lower']<.98 for name in eligible)
    result=dict(scene=scene,safety_layer=layer,planned=len(rows),qualified=len(eligible),controller_rates=rates,informative=informative,
        reason='Qualified baseline lower success <98%' if informative else 'All qualified lower success >=98%; only fixed correction' if eligible else 'No qualified controller')
    write(HERE/f'BASELINE_USABILITY_{layer}_{scene}.json',result)
    if layer=='distributed':write(HERE/f'BASELINE_USABILITY_{scene}.json',result)
    write(HERE/f'{stage}_RESULT_SOURCES.json',sources)
    print(layer,scene,'baseline informative',informative,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='capability':capability()
    else:baseline(*sys.argv[1:])
