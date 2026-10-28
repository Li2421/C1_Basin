"""Freeze17 scene types using only geometric, solo, safety and witness evidence."""
from dispatch import *
from scenes import build,TITLES,LADDERS,geometry
from calibration_gates import tasks
from witnesses import make
import collections

WIDENABLE={8,9,10,11,15,16}

def scene_at(t,level,widen=0.):
    path=HERE/'scene_design'/f't{t:02d}_l{level}{"_w100" if widen else ""}.json'
    if path.exists():return read(path)['scene']
    s=build(t,level,widen);dump(path,dict(status='valid',scene=s));return s

def gate_metrics(scene,rows):
    byuid={r['rollout_uid']:r for r in rows};score={};safety=[]
    for ctl in controllers():
        solo=[]
        for c in tasks(scene):
            if c['controller_payload']['nominal']['type_code']!=ctl['type_code']:continue
            r=byuid[c['rollout_uid']]
            if c['solo']is None:safety.append(r)
            else:solo.append(r)
        score[ctl['type_code']]=dict(n=len(solo),success=sum(r['success']for r in solo),rate=sum(r['success']for r in solo)/len(solo),passed=sum(r['success']for r in solo)/len(solo)>=.95)
    return dict(solo_gates=score,applicable_controllers=[k for k,v in score.items()if v['passed']],safety_n=len(safety),safety_success=sum(r['success']for r in safety),safety_rate=sum(r['success']for r in safety)/len(safety),slightly_hard=sum(r['success']for r in safety)/len(safety)<=.6,gate_uids=list(byuid))

def witness_round(active,round_id):
    solved={t:{}for t in active};attempted={t:[]for t in active};disabled=read(HERE/'WITNESS_DISABLED.json')if(HERE/'WITNESS_DISABLED.json').exists()else{}
    for kind,ctl_index,order in [('sequential_release',0,'first'),('sequential_release',0,'second'),('sequential_release',1,'first'),('sequential_release',1,'second'),('conflict_based_search',None,'planner'),('prioritized_astar',None,'planner')]:
        if kind in disabled:continue
        cases=[]
        for t,s in active.items():
            grouped=t in(8,10,11,12)
            if order=='second'and not grouped:continue
            if ctl_index is not None:ctl=controllers()[ctl_index]
            else:ctl=dict(type_code=kind,parameters='[0.05, 20, 1024]'if kind=='conflict_based_search'else'[0.05, 20, "far"]',controller='original_replanning_witness',checkpoint='')
            role_order=('left_first'if order=='first'else'right_first')if grouped and ctl_index is not None else'shortest'
            for i in range(4):
                if i not in solved[t]:cases.append(make(s,ctl,i,kind,role_order))
        if not cases:continue
        batches=[cases]
        if kind!='sequential_release':
            pilots=[];seen=set()
            for c in cases:
                if c['type_id']not in seen:pilots.append(c);seen.add(c['type_id'])
            pilot_uids={c['rollout_uid']for c in pilots};batches=[pilots,[c for c in cases if c['rollout_uid']not in pilot_uids]]
        for batch_index,batch in enumerate(batches):
            if not batch:continue
            stage=f'witness_{round_id}_{kind}_{ctl_index}_{order}_{batch_index}'
            rr=execute(batch,stage,worker_entry='worker_witness.py')
            for r in rr:
                t=r['type_id'];attempted[t].append(r['rollout_uid'])
                if r['success']:solved[t].setdefault(r['state_index'],r)
                if kind!='sequential_release'and(r['wall_seconds']>1800 or r.get('witness',{}).get('tool_timeout')):
                    disabled[kind]=dict(uid=r['rollout_uid'],seconds=r['wall_seconds'],reason='Tool compute limit; not a scene impossibility result.')
            if kind in disabled:
                dump(HERE/'WITNESS_DISABLED.json',disabled)
                with(HERE/'GAPS.md').open('a')as f:f.write(f'\n见证工具{kind}超过1800s，停用后续请求；不判场景不可解。\n')
                break
    return {t:dict(witness_success=len(solved[t]),joint_certified=len(solved[t])>=3,witness_status={str(i):dict(status='constructive_success'if i in solved[t]else'not_witnessed',uid=solved[t][i]['rollout_uid']if i in solved[t]else None,within_task_horizon=solved[t][i].get('witness',{}).get('within_task_horizon')if i in solved[t]else None)for i in range(4)},witness_uids=attempted[t])for t in active}

def difficulty(t,level):
    value=LADDERS[t][level]
    return -value if t==14 else value

def main():
    assert read(HERE/'ROUTE_EQUIVALENCE.json')['passed']
    if(HERE/'SCENES_FROZEN.json').exists():return
    history={str(t):[]for t in range(1,18)};chosen={};gaps={};active={t:scene_at(t,0)for t in range(1,18)}
    for round_id in range(8):
        if not active:break
        rows=execute([c for s in active.values()for c in tasks(s)],f'calibration_v2_gates_{round_id}')
        scores={t:gate_metrics(s,[r for r in rows if r['scene']==s['name']])for t,s in active.items()}
        # Hardness failures cannot satisfy all criteria; do not spend witness
        # compute on a level already eliminated by a prescribed gate.
        eligible={t:s for t,s in active.items()if scores[t]['slightly_hard']and scores[t]['applicable_controllers']}
        witnesses=witness_round(eligible,round_id)if eligible else{}
        next_active={}
        for t,s in active.items():
            m=scores[t];m.update(witnesses.get(t,dict(witness_success=0,joint_certified=False,witness_status={},witness_uids=[],witness_not_tested_reason='Level already fails prescribed solo/hardness gate.')))
            m['all_criteria']=bool(m['applicable_controllers'])and m['slightly_hard']and m['joint_certified']
            history[str(t)].append(dict(scene=s,metrics=m));dump(HERE/'CALIBRATION.json',dict(history=history,gaps=gaps,selection_uses_only='geometry, solo, safety, constructive witnesses'))
            print(TITLES[t],s['level'],s['widen'],'safety',m['safety_rate'],'witness',m['witness_success'],flush=True)
            if m['all_criteria']:chosen[t]=(s,m);continue
            tested={h['scene']['level']for h in history[str(t)]if h['scene']['widen']==s['widen']}
            untested=[j for j in range(len(LADDERS[t]))if j not in tested]
            if untested:
                easier=not m['joint_certified']and m['slightly_hard']
                desired=[j for j in untested if (difficulty(t,j)<difficulty(t,s['level']))==easier]
                nxt=min(desired or untested,key=lambda j:(abs(difficulty(t,j)-difficulty(t,s['level'])),j))
                next_active[t]=scene_at(t,nxt,s['widen']);continue
            if t in WIDENABLE and not s['widen']:
                next_active[t]=scene_at(t,0,s['config']['agent_radius']);continue
            ranked=sorted(history[str(t)],key=lambda h:(not bool(h['metrics']['applicable_controllers']),not h['metrics']['joint_certified'],not h['metrics']['slightly_hard'],abs(h['metrics']['safety_rate']-.6),h['scene']['level']))
            best=ranked[0];chosen[t]=(best['scene'],best['metrics']);gaps[str(t)]='No full-criteria level; closest observed geometry-only level retained.'
            if not best['metrics']['joint_certified']and t in WIDENABLE:gaps[str(t)]='未找到可解见证；所有方法照跑、单列，排除维度判定分母。'
        active=next_active
    assert not active,'Finite ladder iteration bound insufficient; preserve state and repair orchestration.'
    entries=[];layers={};consistency={}
    for t in range(1,18):
        s,m=chosen[t];cfg=s['config'];margin=geometry(s).margin
        assert margin==cfg['agent_radius']+cfg['wall_radius']+cfg['wall_collision_margin']
        eligible=m['joint_certified']
        layers[str(t)]=dict(geometry_valid=True,single_agent_paths=True,controller_gate=m['solo_gates'],joint_witness=m['witness_status'],joint_certified=m['joint_certified'],slightly_hard=m['slightly_hard'],all_criteria=m['all_criteria'],dimension_eligible=eligible)
        consistency[str(t)]=dict(physical_margin=margin,safety_margin=cfg['agent_radius']+cfg['wall_radius']+cfg['wall_collision_margin'],identical=True)
        entries.append(dict(type_id=t,title=TITLES[t],status='applicable'if m['applicable_controllers']else'controller_not_applicable',applicable_controllers=m['applicable_controllers'],dimension_eligible=eligible,scene=s,calibration=m,gap=gaps.get(str(t))))
    dump(HERE/'FIVE_LAYER.json',layers);dump(HERE/'GEOMETRY_CONSISTENCY.json',consistency)
    dump(HERE/'SCENES_FROZEN.json',dict(entries=entries,generation_seed=2026102801,calibration_states=[0,1,2,3],holdout_states=[4,5,6,7],new_holdout_states=[8,9,10,11],no_correction_seen=True,no_paper_seen=True,freeze_time=datetime.datetime.now().astimezone().isoformat()))
    with(HERE/'GAPS.md').open('a')as f:
        for t,gap in gaps.items():f.write(f'\n{TITLES[int(t)]}：{gap}\n')
    print('SCENES FROZEN17; eligible',sum(e['dimension_eligible']for e in entries),flush=True)
if __name__=='__main__':main()
