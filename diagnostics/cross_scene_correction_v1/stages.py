"""Capability gates, measured pilot, and full fixed-action tables with exact reuse."""
from common import *
from scenes import make_new
from history import key as history_key
from queueing import build
import numpy as np


def scene_list():return read(HERE/'SCENES.json')


def capability_cases():
    pool=read_evidence(V2/'LEARNING_SPLIT_REVISED_20261021_lit_narrow_passage.json')['training_controllers']
    cases=[]
    for scene in scene_list():
        if scene['family']=='literature':continue
        geometry=read(HERE/'geometry'/f"{scene['name']}.json")
        assert all(r['eligible']for r in geometry['states']), ('INVALID_STATES_RETAINED',scene['name'])
        for ctl in pool:
            for state in scene['states']:
                for i in range(state['N']):
                    solo=dict(state,N=1,parent_N=state['N'],agent=i,parent_positions=state['positions'],
                        positions=[state['positions'][i]],goals=[state['goals'][i]],velocities=[state['velocities'][i]],
                        index=state['index']*100+i)
                    for seed in (0,1):cases.append(make_new(scene,ctl,solo,actions()[0],seed))
    return cases


def source_path(stage,c):
    own=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
    if own.exists():return own
    if c.get('reuse_source'):return Path(c['reuse_source'])
    # Pilot and capability records remain immutable when reused by later stages.
    for folder in ('pilot','pilot_tail','capability','main','extrapolation'):
        p=HERE/folder/'raw'/f"{c['rollout_uid']}.json"
        if p.exists():return p
    cached=HERE/'reused_raw'/f"{c['rollout_uid']}.json"
    if cached.exists():return cached
    with dbm.connect(True)as db:
        rows=db.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf USING(source_uid) WHERE rs.rollout_uid=?',
                        (c['rollout_uid'],)).fetchall()
    for row in rows:
        journal=Path(row['path'])
        if not journal.exists():continue
        with journal.open()as f:
            result=next((json.loads(line)for i,line in enumerate(f,1)if i==row['source_line']),None)
        if result is None or result.get('rollout_uid')!=c['rollout_uid']:continue
        if result.get('trace_path'):
            raw=Path(result['trace_path']).parent.parent/'raw'/f"{c['rollout_uid']}.json"
            if raw.exists():return raw
        write(cached,result)
        write(HERE/'reused_raw_provenance'/f"{c['rollout_uid']}.json",dict(source=str(journal),source_sha256=evidence(journal),
            line=row['source_line'],record_hash=uid('record',result),new_physical_steps=0))
        return cached
    raise FileNotFoundError(c['rollout_uid'])


def read_result(stage,c):
    path=source_path(stage,c);r=read(path)
    if not path.resolve().is_relative_to(HERE):
        evidence(path)
    assert r['rollout_uid']==c['rollout_uid']
    return r


def capability_select():
    pool=read(V2/'LEARNING_SPLIT_REVISED_20261021_lit_narrow_passage.json')['training_controllers']
    cases=read(HERE/'capability_CASES.json');table={}
    for c in cases:
        r=read_result('capability',c);name=c['state']['scene'];ctl=c['controller_payload']['nominal']['controller']
        row=table.setdefault(name,{}).setdefault(ctl,dict(total=0,success=0,unknown=0,collision=0,timeout=0))
        row['total']+=1
        for k in ('success','collision','timeout'):row[k]+=int(r[k])
        row['unknown']+=int(r['numerical_failure'])
    selected={}
    for s in scene_list():
        name=s['name']
        if s['family']=='literature':selected[name]=s['training_controllers'];continue
        chosen={}
        for kind in TYPES:
            chosen[kind]=None
            for ctl in pool:
                if ctl['type_code']!=kind:continue
                row=table[name][ctl['controller']];row['rate']=row['success']/row['total']
                row['upper_rate']=(row['success']+row['unknown'])/row['total']
                row['passed']=row['rate']>=.95
                if row['passed']and chosen[kind]is None:chosen[kind]=ctl
        selected[name]=chosen
    write(HERE/'CAPABILITY_TABLE.json',table)
    write(HERE/'SELECTED_CONTROLLERS.json',selected)
    lines=['# 能力门\n','每个新场景的全部标签状态×每台机器人单独×种子0与1；阈值95%，unknown按失败。每类按v2池顺序取第一个通过实例。\n',
           '| 场景 | 实例 | 成功/总数 | unknown | 通过 |\n|---|---|---:|---:|---|\n']
    for name,rows in table.items():
        for ctl,row in rows.items():lines.append(f"| {name} | {ctl} | {row['success']}/{row['total']} | {row['unknown']} | {row['passed']} |\n")
    (HERE/'CAPABILITY_TABLE.md').write_text(''.join(lines))
    return selected


def pilot_cases():
    selected=read(HERE/'SELECTED_CONTROLLERS.json');out=[]
    for s in scene_list():
        if s['family']in ('literature','extrapolation'):continue
        for ctl in selected[s['name']].values():
            if ctl is None:continue
            for state in s['states'][:2]:
                for a in (actions()[0],actions()[14]):out.append(make_new(s,ctl,state,a))
    return out


def pilot_cost():
    from schedule import quota,jobs
    table={}
    for c in read(HERE/'pilot_CASES.json'):
        r=read_result('pilot',c);key=c['state']['scene']+'|'+c['controller_payload']['nominal']['type_code']
        table.setdefault(key,[]).append(float(r['elapsed_seconds']))
    measured={k:dict(n=len(v),mean_seconds=float(np.mean(v)),max_seconds=max(v),all_seconds=v)for k,v in table.items()}
    scenes={s['name']:s for s in scene_list()};total=0.
    for key,m in measured.items():
        scene=scenes[key.split('|')[0]];m['estimated_main_cpu_hours']=len(scene['states'])*49*m['mean_seconds']/3600;total+=m['estimated_main_cpu_hours']
    reference_workers=max(1,quota(jobs()))
    write(HERE/'PILOT_TIMINGS.json',dict(per_scene_controller=measured,estimated_new_main_cpu_hours=total,
        reference_workers=reference_workers,estimated_new_main_wall_hours=total/reference_workers,
        caveat='Two states by two candidates; success/timeout and planning variance can change costs. Excludes capability and extrapolation.'))
    lines=['# 计时试跑\n',f'新场景主标签估计 {total:.1f} CPU小时；按当前可用{reference_workers}并发约 {total/reference_workers:.1f} 小时。仅基于2状态×2候选实测，不含能力门、旧场景补缺和外推，超时及规划尾部会增大误差。撤销及重提的未完成尝试开销另计；实际并发随他人CPU/内存需求调整。\n',
           '| 场景/控制器类型 | 局数 | 平均秒 | 最长秒 | 主标签CPU小时估计 |\n|---|---:|---:|---:|---:|\n']
    for k,m in measured.items():lines.append(f"| {k} | {m['n']} | {m['mean_seconds']:.2f} | {m['max_seconds']:.2f} | {m['estimated_main_cpu_hours']:.2f} |\n")
    (HERE/'PILOT_COST.md').write_text(''.join(lines))


def old_case(scene,ctl,state,action,index,templates):
    # Exact reimplementation of case_factory/ceiling_circle without their evidence writes.
    sample=make_case(state,'flow',action['key'],{},dict(nominal=ctl),0,action['eta_a'])
    existing=index.get(history_key(sample))
    if existing:return copy.deepcopy(existing)
    template=templates[canonical([scene['name'],ctl['controller'],state['N'],state['index']])]
    payload={k:v for k,v in template['controller_payload'].items()if k not in ('method','eta_b','theta')}
    if any(action['eta_a']):
        payload.update(correction_semantics='contact_conflict_relative_gate_20261021',
            correction_revision_receipt_sha256=evidence(V2/'CORRECTION_REVISION.receipt.json'),
            correction_basis='orthoflow3_local_contact_gated',arrived_flag_broadcast=True)
    c=make_case(state,'flow',action['key'],template['scenario_payload'],payload,0,action['eta_a'],template['state_provenance'])
    c['rng_state_uid']=template.get('rng_state_uid',template['state_uid'])
    return c


def main_cases():
    selected=read(HERE/'SELECTED_CONTROLLERS.json');index={history_key(c):c for c in read(HERE/'HISTORY_CASES.json')}
    templates=read_evidence(SOURCE/'HISTORY_TEMPLATES.json');out=[]
    for scene in scene_list():
        if scene['family']=='extrapolation':continue
        for ctl in selected[scene['name']].values():
            if ctl is None:continue
            for state in scene['states']:
                for a in actions():
                    out.append(old_case(scene,ctl,state,a,index,templates) if scene['family']=='literature'else make_new(scene,ctl,state,a))
    return out


def prepare(stage):
    assert read(HERE/'REPLAY_CHECK.json')['passed']
    assert read(HERE/'WRITE_PATH_AUDIT.json')['passed']
    assert read(HERE/'EXTENSION_UNIT_TESTS.json')['passed']
    functions={'capability':capability_cases,'pilot':pilot_cases,'main':main_cases}
    cases=functions[stage]();register(stage,cases);build(stage)


if __name__=='__main__':
    prepare(sys.argv[1])
