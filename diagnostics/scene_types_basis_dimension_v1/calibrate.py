"""Freeze difficulty using only geometry, solo gates, safety and A* witnesses."""
from dispatch import *
from scenes import *

def astar():
    # The original priority tracker, original space-time kernel and original
    # reservation certificates. This is a witness, not a nominal replacement.
    return dict(type_code='prioritized_astar',parameters='[0.05, 100000, "far"]',controller='original_priority_spacetime_astar_witness',checkpoint='')

def tasks(scene):
    cs=[]
    for ctl in controllers():
        for i in range(4):
            cs.append(case(scene,ctl,i,purpose='calibration_safety'))
            for j in range(scene['config']['num_agents']):cs.append(case(scene,ctl,i,solo=j,purpose='single_robot_gate'))
    for i in range(4):cs.append(case(scene,astar(),i,purpose='joint_feasibility_witness'))
    return cs

def evaluate(scene,rows):
    ctls={}
    for ctl in controllers():
        name=ctl['type_code'];rr=[r for r in rows if r['controller']==name and r['solo']is not None]
        rate=sum(r['success']for r in rr)/len(rr)if rr else 0.
        ctls[name]=dict(n=len(rr),success=sum(r['success']for r in rr),rate=rate,passed=rate>=.95,numerical_unknown=sum(r['numerical_failure']for r in rr))
    applicable=[k for k,v in ctls.items()if v['passed']]
    saf=[r for r in rows if r['purpose']=='calibration_safety'and r['controller']in applicable]
    wit=[r for r in rows if r['purpose']=='joint_feasibility_witness']
    safety_rate=sum(r['success']for r in saf)/len(saf)if saf else None
    w=sum(r['success']for r in wit)
    return dict(type_id=scene['type_id'],level=scene['level'],widen=scene['widen'],solo_gates=ctls,applicable_controllers=applicable,
        safety_n=len(saf),safety_success=sum(r['success']for r in saf),safety_rate=safety_rate,slightly_hard=safety_rate is not None and safety_rate<=.6,
        witness_n=len(wit),witness_success=w,joint_certified=w>=3,
        all_criteria=bool(applicable)and safety_rate is not None and safety_rate<=.6 and w>=3,
        raw_rollout_uids=[r['rollout_uid']for r in rows],witness_status={str(r['state_index']):('constructive_success'if r['success']else'uncertified')for r in wit})

def main():
    assert read(HERE/'ROUTE_EQUIVALENCE.json')['passed']
    frozen=HERE/'SCENES_FROZEN.json'
    if frozen.exists():print('SCENES ALREADY FROZEN',flush=True);return
    build_all();ctls=controllers();active={};history={str(t):[]for t in range(1,18)};chosen={};gaps={}
    for t in range(1,18):
        value=read(HERE/'scene_design'/f't{t:02d}_l0.json')
        if value['status']!='valid':gaps[str(t)]=value['error'];continue
        active[t]=value['scene']
    for round_index in range(5):
        if not active:break
        cs=[c for s in active.values()for c in tasks(s)]
        if len(list((HERE/'attempts').glob('*.json')))+len(cs)>6000:
            for t,s in active.items():chosen[t]=s;gaps[str(t)]='Calibration cap reached before all ladder conditions certified.'
            break
        rows=execute(cs,f'calibration_{round_index}')
        next_active={}
        for t,s in active.items():
            rr=[r for r in rows if r['scene']==s['name']];score=evaluate(s,rr)
            history[str(t)].append(dict(scene=s,metrics=score));dump(HERE/'CALIBRATION.json',dict(history=history,gaps=gaps,criterion='Predeclared solo>=95%, safety<=60%, A*>=3/4; no correction labels.'))
            print('CALIBRATION',t,s['level'],s['widen'],score['applicable_controllers'],score['safety_rate'],'witness',score['witness_success'],flush=True)
            if not score['applicable_controllers']:
                if not s['widen']:
                    try:next_active[t]=build(t,s['level'],.1)
                    except Exception as exc:gaps[str(t)]='Both nominal gates failed; prescribed width relaxation unavailable: '+str(exc)
                else:gaps[str(t)]='Both nominal gates failed after the one prescribed +0.5d widening.'
                if t not in next_active:chosen[t]=s
                continue
            if score['all_criteria']:chosen[t]=s;continue
            tested={h['scene']['level']for h in history[str(t)]}
            if not score['joint_certified']:
                # The explicit last/easier entry is used first for failed witnesses.
                nxt=len(LADDERS[t])-1
                if nxt in tested:nxt=next((k for k in range(len(LADDERS[t]))if k not in tested),None)
            else:nxt=next((k for k in range(s['level']+1,len(LADDERS[t]))if k not in tested),None)
            if nxt is not None:
                value=read(HERE/'scene_design'/f't{t:02d}_l{nxt}.json')
                if value['status']=='valid':next_active[t]=value['scene'];continue
                gaps[str(t)]=value['error']
            # Closest ladder: prefer an applicable gate, then certified witness,
            # then safety difficulty nearest the prescribed threshold.
            ranked=sorted(history[str(t)],key=lambda h:(not bool(h['metrics']['applicable_controllers']),not h['metrics']['joint_certified'],abs((h['metrics']['safety_rate']or 0.)-.6),h['scene']['level']))
            chosen[t]=ranked[0]['scene'];gaps.setdefault(str(t),'No ladder satisfies every condition; closest tested level frozen, evidence layers remain explicit.')
        active=next_active
    for t,s in active.items():chosen.setdefault(t,s);gaps.setdefault(str(t),'Calibration round cap reached; selected closest observed level, incomplete gate certification.')
    # State provenance / novelty: compare complete coordinates against historical
    # states, independent of scenario IDs or critic encoding.
    hashes=set()
    with db.connect(True)as con:
        for row in con.execute('SELECT physical_state_json FROM state WHERE physical_state_json IS NOT NULL'):
            hashes.add(hashlib.sha256(row[0].encode()).hexdigest())
    duplicate=[]
    for t,s in chosen.items():
        for state in s['states']:
            physical={k:state[k]for k in ('positions','velocities','goals')}
            # Own newly registered calibration states are expected exact matches;
            # historical duplicates are checked through experiment provenance.
            with db.connect(True)as con:
                rows=con.execute('SELECT provenance_json FROM state WHERE physical_state_json=?',(db.canonical(physical),)).fetchall()
            if any(json.loads(x[0]).get('experiment')!=EXP for x in rows):duplicate.append((t,state['index']))
    assert not duplicate,('Historical physical state duplicates',duplicate)
    layers={};consistency={};entries=[]
    for t in range(1,18):
        if t not in chosen:
            layers[str(t)]=dict(geometry_valid=False,single_agent_paths=False,joint_feasible='not_evaluated',controller_gate='not_applicable',deadlock_resolution='not_evaluated',gap=gaps.get(str(t)))
            entries.append(dict(type_id=t,title=TITLES[t],status='geometry_gap',gap=gaps.get(str(t))));continue
        s=chosen[t];h=next((h for h in history[str(t)]if h['scene']['definition_sha256']==s['definition_sha256']),None)
        m=h['metrics']if h else dict(applicable_controllers=[])
        applicable=m['applicable_controllers'];layers[str(t)]=dict(geometry_valid=True,single_agent_paths=True,controller_gate=m.get('solo_gates'),joint_feasible=m.get('witness_status',{}),deadlock_resolution='not_evaluated',all_calibration_conditions=m.get('all_criteria',False))
        physical_margin=geometry(s).margin;cfg=s['config'];safety_margin=cfg['agent_radius']+cfg['wall_radius']+cfg['wall_collision_margin']
        assert physical_margin==safety_margin
        consistency[str(t)]=dict(agent_radius=cfg['agent_radius'],physical_wall_radius=4/600,safety_wall_radius=cfg['wall_radius'],physical_margin=physical_margin,safety_margin=safety_margin,cbf_additional_separation_buffer=RN.HardProjectionConfig().separation_buffer,identical=True)
        entries.append(dict(type_id=t,title=TITLES[t],status='applicable'if applicable else'controller_not_applicable',applicable_controllers=applicable,scene=s,calibration=m,gap=gaps.get(str(t))))
    dump(HERE/'GEOMETRY_CONSISTENCY.json',consistency);dump(HERE/'FIVE_LAYER.json',layers)
    dump(frozen,dict(entries=entries,generation_seed=SEED,calibration_states=[0,1,2,3],holdout_states=[4,5,6,7],reserved_unused=[8,9,10,11],no_correction_seen=True,novelty_check='Exact full physical states; no prior-experiment coordinate duplicate.',freeze_time=datetime.datetime.now().astimezone().isoformat()))
    (HERE/'GAPS.md').write_text('# 证据缺口\n\n'+''.join(f'- {TITLES[int(t)]}：{g}\n'for t,g in gaps.items())+'\n某控制器不适用时，留出分母仍按预登记 8 个案例显示，缺失项不会用成功填补；足够类型的 6/8 门槛不改。\n')
    print('SCENES FROZEN',len(chosen),'physical types;',sum(x['status']=='applicable'for x in entries),'applicable',flush=True)

if __name__=='__main__':main()
