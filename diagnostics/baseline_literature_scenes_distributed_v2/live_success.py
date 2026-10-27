"""Observed safety-only rates; partial coverage is explicitly labelled."""
from experiment import *
from result_inventory import qualification
import collections
import csv

def main():
    counts=read(HERE/'STUDY_DESIGN.json')['main_state_counts'];scenes={};controllers=[]
    for scene in NAMES:
        q={r['controller'] for r in qualification(scene) if r['qualified']}
        all_counts=collections.Counter();qualified_counts=collections.Counter();groups=collections.defaultdict(list)
        for path in (HERE/f'distributed_baseline_{scene}'/'raw').glob('*.json'):
            r=read(path);ctl=r['case']['controller_payload']['nominal']
            values=dict(completed=1,success=r['success'],unknown=r['numerical_failure'],collision=r['collision'],timeout=r['timeout'])
            all_counts.update(values)
            if ctl['controller'] in q:qualified_counts.update(values)
            groups[(ctl['controller'],ctl['type_code'])].append(values)
        scenes[scene]=dict(qualified=dict(qualified_counts),all48=dict(all_counts),qualified_controllers=len(q),expected_qualified=len(q)*counts[scene]*2,expected_all=48*counts[scene]*2,
            complete=all_counts['completed']==48*counts[scene]*2)
        for (name,type_code),rows in sorted(groups.items()):
            success=sum(r['success'] for r in rows);unknown=sum(r['unknown'] for r in rows)
            controllers.append(dict(scene=scene,controller=name,type=type_code,qualified=name in q,completed=len(rows),planned=counts[scene]*2,success=success,unknown=unknown,rate_lower=success/len(rows),rate_upper=(success+unknown)/len(rows),complete=len(rows)==counts[scene]*2))
    result=dict(utc=utc(),layer='distributed',eta='zero',weighting='Observed episode pooling; not final type-equal bootstrap',scenes=scenes)
    write(HERE/'LIVE_SAFETY_SUCCESS.json',result);write(HERE/'LIVE_CONTROLLER_SAFETY_SUCCESS.json',dict(utc=utc(),rows=controllers))
    with (HERE/'LIVE_CONTROLLER_SAFETY_SUCCESS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(controllers[0]) if controllers else ['scene']);writer.writeheader();writer.writerows(controllers)
    return result

if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
