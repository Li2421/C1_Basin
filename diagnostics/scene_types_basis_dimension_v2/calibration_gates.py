"""Geometry-only base-level calibration: no correction or paper outcome input."""
from dispatch import *
from scenes import TITLES

def tasks(scene):
    result=[]
    for ctl in controllers():
        for i in range(4):
            result.append(case(scene,ctl,i,purpose='calibration_safety'))
            for j in range(scene['config']['num_agents']):result.append(case(scene,ctl,i,solo=j,purpose='single_robot_gate'))
    return result

def main():
    assert read(HERE/'ROUTE_EQUIVALENCE.json')['passed']
    scenes=[read(HERE/'scene_design'/f't{t:02d}_l0.json')['scene']for t in range(1,18)]
    cases=[c for s in scenes for c in tasks(s)]
    assert all(not any(c['spec']['eta'])and not c['spec']['coefficients']for c in cases)
    rows=execute(cases,'calibration_base_gates')
    summary=[]
    for s in scenes:
        for ctl in controllers():
            rr=[r for r in rows if r['scene']==s['name']and r['controller']==ctl['type_code']]
            solo=[r for r in rr if r['solo']is not None];team=[r for r in rr if r['solo']is None]
            summary.append(dict(type_id=s['type_id'],title=TITLES[s['type_id']],controller=ctl['type_code'],solo_n=len(solo),solo_success=sum(r['success']for r in solo),solo_rate=sum(r['success']for r in solo)/len(solo),safety_n=len(team),safety_success=sum(r['success']for r in team),raw_uids=[r['rollout_uid']for r in rr]))
    dump(HERE/'CALIBRATION_BASE_GATES.json',summary)
    print('Base solo/safety calibration complete',len(rows),flush=True)
if __name__=='__main__':main()
