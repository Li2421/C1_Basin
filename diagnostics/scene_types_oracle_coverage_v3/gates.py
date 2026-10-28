"""Reuse exact old solo gates; only repair absent evidence and new geometry."""
from dispatch import *
from scenes import MODIFIED

def main():
    p=HERE/'SOLO_GATES.json'
    if p.exists()and read(p).get('complete'):return read(p)
    scenes=read(HERE/'SCENES.json')['entries']
    old=read(ROOT/'diagnostics/scene_types_basis_dimension_v1/CALIBRATION.json')['history']
    inherited={}
    for rows in old.values():
        for h in rows:
            s=h['scene']
            if s['widen']==0.:inherited[(s['type_id'],s['level'])]=h['metrics']['solo_gates']
    cases=[];requirements=[]
    for e in scenes:
        t,level=e['type_id'],e['level'];s=e['scene']
        if t not in MODIFIED and (t,level)in inherited:continue
        states=list(range(8))if t in MODIFIED else list(range(4))
        requirements.append(dict(type_id=t,level=level,states=states,
                                 reason='new prescribed geometry'if t in MODIFIED else'missing exact level in v1 solo records'))
        for ctl in controllers():
            for si in states:
                for agent in range(s['config']['num_agents']):
                    cases.append(case(s,ctl,si,solo=agent,purpose='v3_single_robot_gate'))
    dump(HERE/'SOLO_GATE_REQUIREMENTS.json',requirements)
    rows=execute(cases,'solo_gates',worker_entry='worker_recorded.py')
    entries=[]
    for e in scenes:
        t,level=e['type_id'],e['level']
        if t not in MODIFIED and (t,level)in inherited:
            gates=inherited[(t,level)];source='v1_calibration_exact_level'
        else:
            gates={};source='v3_missing_evidence_rollouts'
            for ctl in controllers():
                rr=[r for r in rows if r['scene']==e['scene']['name']and r['controller']==ctl['type_code']]
                expected=e['scene']['config']['num_agents']*(8 if t in MODIFIED else 4)
                rate=sum(r['success']for r in rr)/expected
                gates[ctl['type_code']]=dict(n=len(rr),expected=expected,success=sum(r['success']for r in rr),
                    rate=rate,passed=len(rr)==expected and rate>=.95,
                    numerical_unknown=sum(r['numerical_failure']for r in rr),rollout_uids=[r['rollout_uid']for r in rr])
        entries.append(dict(type_id=t,level=level,gates=gates,source=source,
                            applicable_controllers=[name for name,g in gates.items()if g['passed']]))
    complete=all(g.get('n')==g.get('expected',g.get('n'))for e in entries for g in e['gates'].values())
    result=dict(complete=complete,entries=entries,threshold=.95,no_difficulty_selection=True)
    dump(p,result);return result

if __name__=='__main__':main()
