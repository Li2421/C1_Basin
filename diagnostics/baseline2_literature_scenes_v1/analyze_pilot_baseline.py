from experiment import *
from phase_reports import records
from failure_analysis import diagnose,census
import numpy as np
import csv
import collections

def main(scene):
    stage='pilot_safety_'+scene;raw=records(stage);planned=len(read(HERE/f'{stage}_CASES.json'));assert len(raw)==planned
    qualification=[]
    for p in HERE.glob('pilot_capability*_QUALIFICATION.json'):qualification+=read(p)
    q={r['controller']:r for r in qualification if r['scene']==scene};groups=collections.defaultdict(list);rows=[];diagnostic=[];cens=[]
    for x in raw:
        c=x['case'];ctl=c['controller_payload']['nominal'];s=c['state'];groups[ctl['controller']].append(x)
        with np.load(x['trace_path']) as z:
            categories=diagnose(x,z)
            for agent,category in enumerate(categories):rows.append(dict(scene=NAMES[scene],scene_code=scene,N=s['N'],state=s['index'],controller=ctl['controller'],type=ctl['type'],type_code=ctl['type_code'],seed=c['seed'],success=x['success'],collision=x['collision'],timeout=x['timeout'],unknown=x['numerical_failure'],agent=agent,category=category,rollout_uid=x['rollout_uid'],qualified=q.get(ctl['controller'],{}).get('qualified',False)))
            for W in (2.5,5.,10.):
                for tau in (.02,.05,.1):diagnostic.append(dict(rollout_uid=x['rollout_uid'],window_seconds=W,cone_tolerance=tau,categories=diagnose(x,z,W,tau)))
            cens+=census(x,z)
    with (HERE/f'PILOT_SAFETY_BASELINE_{scene}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    write(HERE/f'PILOT_FAILURE_SENSITIVITY_{scene}.json',diagnostic);write(HERE/f'PILOT_GEOMETRY_CENSUS_{scene}.json',cens)
    eligible=[name for name in groups if q.get(name,{}).get('qualified',False)]
    rates={name:np.mean([r['success'] for r in rr]) for name,rr in groups.items()}
    informative=bool(eligible and any(rates[name]<.98 for name in eligible))
    result=dict(scene=scene,episodes=len(raw),planned=planned,qualified=len(eligible),controller_success_rates=rates,informative=informative,
                reason='At least one qualified black-box controller has safety-only success <98%' if informative else 'All qualified controllers >=98%; only fixed correction is informative under the protocol' if eligible else 'No qualified controller',
                category_counts=dict(collections.Counter(r['category'] for r in rows)),census_steps=len(cens))
    write(HERE/f'PILOT_BASELINE_USABILITY_{scene}.json',result)
    print(canonical(result))
if __name__=='__main__':main(sys.argv[1])
