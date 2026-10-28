"""Resumable v3 coverage pipeline; one authoritative merger, sequential eta per case."""
from dispatch import *
from acceleration import accelerated_case
from reuse import requests
from coverage import write,FAMILY_NAMES
from budget import used
import traceback

ORDER=[4,0,1,2,3,5,6]

def wait_for_previous_stage(filename,job_name):
    while not(HERE/filename).exists():
        running=[j for j in jobs()if j['name']==job_name]
        if not running:return False
        status('waiting_'+filename,0,1,'等待已经提交的同一阶段收尾；不启动第二个合并协调程序。')
        time.sleep(10)
    return True

def run():
    assert read(HERE/'ROUTE_EQUIVALENCE.json')['passed']
    if not wait_for_previous_stage('SOLO_GATES.json','stb3_solocoord'):
        from gates import main as gate_main
        gate_main()
    if not wait_for_previous_stage('PREFIX_CHECK.json','stb3_prefixcoord'):
        from prefix import main as prefix_main
        prefix_main()
    from certify import main as certify_main
    certificate=certify_main()
    accelerated=certificate['passed']
    chain=HERE/'ACCELERATION_KEY_CHAIN_EXTENSION.json'
    if not wait_for_previous_stage(chain.name,'stb3_rngchains'):
        try:
            from key_chain_extension import main as key_main
            key_main()
        except Exception:
            accelerated=False
            with(HERE/'GAPS.md').open('a')as f:f.write('\n完整随机数链核对失败；主线禁用加速，使用原始数值路径。\n')
    if chain.exists()and not read(chain)['passed']:accelerated=False
    make=accelerated_case if accelerated else case
    scenes=read(HERE/'SCENES.json')['entries'];menus=read(HERE/'FAMILIES_64.json')['families']
    gate={(e['type_id'],e['level']):e for e in read(HERE/'SOLO_GATES.json')['entries']}
    tasks=[];identities=[]
    for e in scenes:
        for ctl in controllers():
            for si in range(8):
                applicable=ctl['type_code']in gate[(e['type_id'],e['level'])]['applicable_controllers']
                identities.append((e,ctl,si,applicable))
                if applicable:tasks.append(make(e['scene'],ctl,si,purpose='v3_safety_only'))
    baseline_records=requests(tasks,'safety_only');baseline=[]
    for c in tasks:
        r=baseline_records.get(c['rollout_uid'])
        if r:baseline.append(dict(type_id=c['type_id'],level=next(e['level']for e in scenes if e['scene']['name']==c['scene']),
            controller=c['controller_payload']['nominal']['type_code'],state_index=c['state_index'],
            success=r['success'],episode_length=r['episode_length'],outcome=r['outcome'],rollout_uid=r['rollout_uid']))
    dump(HERE/'SAFETY_ONLY.json',dict(cases=baseline,complete=len(baseline)==len(tasks)))
    groups=[]
    for fi in range(7):
        for e,ctl,si,applicable in identities:
            groups.append(dict(family=fi,type_id=e['type_id'],level=e['level'],scene=e['scene']['name'],
                controller=ctl['type_code'],state_index=si,success=False,searched=0,first_success=None,
                queries=[],skipped=not applicable,gap=None))
    write(groups,baseline)
    # Planning the full 64-product may exceed the hard actual budget; do not trim
    # scenes, states or candidates. Execute in the declared order until 90000.
    worst=len(groups)*64+len(tasks)+used()
    if not(HERE/'PLAN_BUDGET.json').exists():
        dump(HERE/'PLAN_BUDGET.json',dict(theoretical_upper_bound=worst,hard_actual_limit=90000,
            policy='Early success and certified reuse save actual rollouts. Stop at the hard cap; leave unrun families as explicit gaps.'))
        if worst>90000:
            with(HERE/'ITERATION_LOG.md').open('a')as f:f.write(f'\n完整笛卡尔积上界约{worst}局，超过90000实际新rollout硬上限；不删场景或候选，按推荐四列、原三列、其余五族顺序执行，到上限保留未完成项。\n')
    scene_map={(e['type_id'],e['level']):e for e in scenes}
    ctl_map={c['type_code']:c for c in controllers()}
    for fi in ORDER:
        members=[g for g in groups if g['family']==fi]
        for ki,sp in enumerate(menus[fi]['candidate_pool']):
            active=[g for g in members if not g['success']and not g['skipped']and not g['gap']]
            if not active:break
            cases=[make(scene_map[(g['type_id'],g['level'])]['scene'],ctl_map[g['controller']],g['state_index'],sp,
                        purpose='v3_oracle_coverage')for g in active]
            stage=f'oracle_f{fi}_k{ki+1:02d}'
            rows=requests(cases,stage)
            labels=[]
            for g,c in zip(active,cases):
                uid=c['rollout_uid'];r=rows.get(uid)
                if r is None:
                    g['gap']='Infrastructure incomplete after retries; not a 64-candidate failure.'
                    continue
                query=dict(index=ki+1,target_query_uid=uid,rollout_uid=r['rollout_uid'],
                    success=r['success'],outcome=r['outcome'],source=r['source'],
                    completion_time=r['completion_time'],episode_length=r['episode_length'],trace=r['trace'])
                g['queries'].append(query);g['searched']=ki+1
                if r['success']:
                    g['success']=True;g['first_success']=ki+1;g['solution']=dict(query,spec=sp)
                labels.append(dict(family=fi,type_id=g['type_id'],level=g['level'],
                    controller=g['controller'],state_index=g['state_index'],**query))
            dump(HERE/f'{stage}_labels.json',labels)
            dump(HERE/'SEARCH_STATE.json',dict(groups=groups,baseline=baseline,active_family=fi,last_candidate=ki+1))
            result=write(groups,baseline)
            status(FAMILY_NAMES[fi],sum(g['success']or g['searched']==64 or g['skipped']for g in members),len(members),
                   f'候选序号 {ki+1}/64；当前找到成功 {sum(g["success"]for g in members)}/784；实际新运行 {used()}/90000。')
            print('COVERAGE',fi,ki+1,'success',sum(g['success']for g in members),'actual_new',used(),flush=True)
        dump(HERE/f'family_{fi}_complete.json',dict(family=fi,complete=all(g['success']or g['searched']==64 or g['skipped']for g in members)))
    result=write(groups,baseline)
    dump(HERE/'COMPLETE.json',dict(complete=result['complete'],actual_new_rollouts=used(),hard_limit=90000,
        family_order=ORDER,report=str(HERE/'REPORT.md'),no_dimension_selection=True,no_paper_methods=True))

def main():
    try:
        # A killed coordinator can leave independent array workers running.
        # Drain those workers before the single official merger recovers them.
        own=os.environ.get('SLURM_JOB_ID')
        while any(j['name'].startswith('stb3_')and j['job_id']!=own for j in jobs()):
            time.sleep(10)
        if any((HERE/'raw').glob('*.json')):
            merge('resume_recovery',[dict(rollout_uid=p.stem)for p in(HERE/'raw').glob('*.json')])
        run()
    except Exception as exc:
        if(HERE/'SEARCH_STATE.json').exists():
            state=read(HERE/'SEARCH_STATE.json');write(state['groups'],state['baseline'])
        dump(HERE/'PIPELINE_ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),actual_new_rollouts=used()))
        with(HERE/'GAPS.md').open('a')as f:f.write('\n协调程序停止：'+repr(exc)+'。保留已完成结果，未完成项不算64次失败。\n')
        # Drain and merge any completed journals; this coordinator is the only writer.
        active=[j for j in jobs()if j['name'].startswith('stb3_')and j['name']not in ('stb3_pipeline',)]
        for j in active:
            if 'rngchains'not in j['name']:subprocess.run(['scancel',j['job_id']],check=True)
        while any(j['name'].startswith('stb3_')and j['name']!='stb3_pipeline'for j in jobs()):
            time.sleep(5)
        raw_cases=[dict(rollout_uid=p.stem)for p in(HERE/'raw').glob('*.json')]
        merge('pipeline_stop',raw_cases)
        dump(HERE/'COMPLETE.json',dict(complete=False,error=repr(exc),actual_new_rollouts=used(),resume='bash diagnostics/scene_types_oracle_coverage_v3/resume.sh'))
        raise

if __name__=='__main__':main()
