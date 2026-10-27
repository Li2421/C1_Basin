"""Sequential registered stages; no test rollout can precede the frozen model."""
from common import *
from requests_new import prepare,pool
from schedule import run,log
import numpy as np,subprocess

def result(stage,c):
    for folder in (stage,'capability','pilot','main'):
        p=HERE/folder/'raw'/f"{c['rollout_uid']}.json"
        if p.exists():return read(p)
    with dbm.connect(True)as db:
        rs=db.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf USING(source_uid)WHERE rs.rollout_uid=?',(c['rollout_uid'],)).fetchall()
    for row in rs:
        p=Path(row['path'])
        if not p.exists():continue
        with p.open()as f:
            r=next((json.loads(line)for i,line in enumerate(f,1)if i==row['source_line']),None)
        if r and r.get('rollout_uid')==c['rollout_uid']:
            evidence(p);assert sha(r['trace_path'])==r['trace_sha256'];return r
    raise FileNotFoundError(c['rollout_uid'])

def capabilities():
    table={}
    for c in read(HERE/'capability_CASES.json'):
        r=result('capability',c);name=c['prospective_scene'];ctl=c['controller_payload']['nominal']['controller']
        row=table.setdefault(name,{}).setdefault(ctl,dict(total=0,success=0,unknown=0,collision=0,timeout=0))
        row['total']+=1
        for k in ('success','collision','timeout'):row[k]+=int(r[k])
        row['unknown']+=int(r['numerical_failure'])
    selected={}
    for s in read(HERE/'SCENES.json'):
        if not s['eligible']:continue
        chosen={kind:None for kind in TYPES}
        for ctl in pool():
            row=table[s['name']][ctl['controller']];row.update(rate=row['success']/row['total'],
                unknown_success_upper=(row['success']+row['unknown'])/row['total'],passed=row['success']/row['total']>=.95)
            if row['passed']and chosen[ctl['type_code']]is None:chosen[ctl['type_code']]=ctl
        selected[s['name']]=chosen
    write(HERE/'CAPABILITY_TABLE.json',table);write(HERE/'SELECTED_CONTROLLERS.json',selected)
    names={s['name']:s['full_name']for s in read(HERE/'SCENES.json')}
    lines=['# 能力门\n\n','全部测试状态、每台机器人单独、种子0与1；unknown计失败，阈值95%。每类按文献线训练池顺序选第一个通过实例，未通过的类型逐场景排除。确定性控制器的第二种子沿用上一轮的逐位确定性证据，记录来源，不新增物理步。\n\n',
        '| 场景 | 控制器实例 | 成功/总数 | unknown | 通过 |\n|---|---|---:|---:|---|\n']
    for n,rows in table.items():
        for ctl,r in rows.items():lines.append(f"| {names[n]}（{n}） | {ctl} | {r['success']}/{r['total']} | {r['unknown']} | {r['passed']} |\n")
    lines.append('\n未过门的类型：\n\n')
    for n,row in selected.items():lines.append(f"- {names[n]}（{n}）："+'、'.join(k for k,c in row.items()if c is None)+'\n')
    (HERE/'CAPABILITY_TABLE.md').write_text(''.join(lines))

def pilot_cost():
    table={};scenes={s['name']:s for s in read(HERE/'SCENES.json')};choices=read(HERE/'FROZEN_CHOICES.json')
    for c in read(HERE/'pilot_CASES.json'):
        r=result('pilot',c);k=c['prospective_scene']+'|'+c['controller_payload']['nominal']['type_code']
        table.setdefault(k,[]).append(float(r['elapsed_seconds']))
    measured={};total=0.
    for k,v in table.items():
        n=k.split('|')[0];h=len(scenes[n]['states'])*len(set(choices[n].values()))*np.mean(v)/3600
        measured[k]=dict(n=len(v),mean_seconds=float(np.mean(v)),max_seconds=max(v),all_seconds=v,estimated_main_cpu_hours=float(h));total+=h
    write(HERE/'PILOT_TIMINGS.json',dict(per_scene_controller=measured,estimated_main_cpu_hours=total,estimated_wall_hours_at17=total/17,
        caveat='Two states x two reference candidates; planner tail costs may differ; capability time excluded.'))
    lines=['# 计时试跑\n\n',f'正式评估估计 {total:.2f} CPU小时；17个worker加1个协调CPU时约 {total/17:.2f} 小时，不含已完成能力门。按2状态×2候选测量，规划尾部与结果时长差异可能增大误差。\n\n',
        '| 场景/控制器类型 | 单局秒数（4局） | 平均秒 | 最长秒 | 预计CPU小时 |\n|---|---|---:|---:|---:|\n']
    for k,r in measured.items():lines.append(f"| {k} | {', '.join(f'{v:.2f}'for v in r['all_seconds'])} | {r['mean_seconds']:.2f} | {r['max_seconds']:.2f} | {r['estimated_main_cpu_hours']:.2f} |\n")
    (HERE/'PILOT_COST.md').write_text(''.join(lines))

def labels():
    scenes=read(HERE/'SCENES.json');chosen=read(HERE/'SELECTED_CONTROLLERS.json');choices=read(HERE/'FROZEN_CHOICES.json');out={}
    from analysis_stats import METHODS
    rows={c['rollout_uid']:c for c in read(HERE/'main_CASES.json')}
    for s in scenes:
        if not s['eligible']:continue
        ctls=[c for c in chosen[s['name']].values()if c is not None]
        keys=[a['key']for a in actions()if a['key']in set(choices[s['name']].values())]
        y=np.zeros((len(ctls),len(s['states']),len(keys)));u=y.copy();collision=y.copy();timeout=y.copy()
        ids={ctl['controller']:i for i,ctl in enumerate(ctls)};st={v['index']:i for i,v in enumerate(s['states'])};ac={a:i for i,a in enumerate(keys)}
        for c in rows.values():
            if c['prospective_scene']!=s['name']:continue
            r=result('main',c);i=ids[c['controller_payload']['nominal']['controller']];j=st[c['state']['index']];k=ac[c['requested_action']]
            y[i,j,k]=int(r['success']);u[i,j,k]=int(r['numerical_failure']);collision[i,j,k]=int(r['collision']);timeout[i,j,k]=int(r['timeout'])
        if not ctls:raise RuntimeError(('NO_CAPABLE_CONTROLLER_TYPES',s['name']))
        out[s['name']]=dict(controllers=[c['controller']for c in ctls],state_indices=list(st),action_keys=keys,
            y=y.tolist(),unknown=u.tolist(),collision=collision.tolist(),timeout=timeout.tolist(),
            method_y=np.stack([y[:,:,ac[choices[s['name']][m]]]for m in METHODS],axis=-1).tolist(),
            method_unknown=np.stack([u[:,:,ac[choices[s['name']][m]]]for m in METHODS],axis=-1).tolist())
    write(HERE/'TEST_LABEL_TABLES.json',out)

def main():
    assert read(HERE/'REPLAY_CHECK.json')['passed'] and read(HERE/'EXECUTION_CHECK.json')['passed']
    sealfile=HERE/'EXECUTION_FROZEN.receipt.json'
    if not sealfile.exists():
        paths=[p for p in HERE.glob('*.py')]+[HERE/'worker.sbatch',HERE/'PROSPECTIVE_EXTENSION_BUILD.json',HERE/'BACKEND_BUILD.json',HERE/'EXECUTION_CHECK.json']
        seal(sealfile,paths,models_frozen_before_all_test_rollouts=True,maximum_total_cpu_shares=18,maximum_gpu_shares=2,
            worker_cpu_shares=17,coordinator_cpu_shares=1,actual_requested_gpu_shares=0,
            no_no_result_age_cancel_rule=True,stop_claiming_before_allocation_end_seconds=3300)
    for stage,finish in [('capability',capabilities),('pilot',pilot_cost),('main',labels)]:
        if not(HERE/f'{stage}_COMPLETE.json').exists():
            if not(HERE/f'{stage}_PREFLIGHT.receipt.json').exists():prepare(stage)
            run(stage);finish();write(HERE/f'{stage}_COMPLETE.json',dict(utc=utc(),stage=stage))
            print('Stage completed',stage,flush=True)
    subprocess.run([sys.executable,'-B',str(HERE/'analysis_stats.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(HERE/'deliver.py')],cwd=ROOT,check=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(HERE/'COORDINATOR_FATAL.json',dict(utc=utc(),type=type(e).__name__,message=str(e),traceback=__import__('traceback').format_exc()))
        for r in __import__('schedule').jobs():
            if r['name'].startswith('xsp_'):__import__('schedule').cancel(r['id'],True)
        raise
