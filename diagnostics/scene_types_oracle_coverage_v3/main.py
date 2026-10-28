"""Resumable v3 coverage pipeline; one authoritative merger, sequential eta per case."""
from dispatch import *
from acceleration import accelerated_case
from reuse import requests,cached,flush
from coverage import write,FAMILY_NAMES
from budget import used
import traceback

ORDER=[4,0,1,2,3,5,6]

def consume(g,c,r,index):
    assert index==g['searched']+1 and not g['success']
    query=dict(index=index,target_query_uid=c['rollout_uid'],rollout_uid=r['rollout_uid'],
        success=r['success'],outcome=r['outcome'],source=r['source'],
        completion_time=r['completion_time'],episode_length=r['episode_length'],trace=r['trace'])
    g['queries'].append(query);g['searched']=index
    if r['success']:
        g['success']=True;g['first_success']=index;g['solution']=dict(query,spec=c['spec'])
    return dict(family=g['family'],type_id=g['type_id'],level=g['level'],
        controller=g['controller'],state_index=g['state_index'],**query)

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
    saved=read(HERE/'SEARCH_STATE.json')if(HERE/'SEARCH_STATE.json').exists()else{}
    if saved:
        assert len(saved['groups'])==len(groups)
        groups=saved['groups']
    progress=saved.get('completed_batches',{})
    pending=saved.get('pending_stage')
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
    if pending:
        # Reuse the exact pending plan, even if some/all workers completed
        # before coordinator recovery. Never regenerate that batch's frontier.
        group_map={(g['family'],g['type_id'],g['level'],g['controller'],g['state_index']):g for g in groups}
        frontier=[]
        for item in read(HERE/f'{pending}_frontier.json'):
            g=group_map[(item['family'],item['type_id'],item['level'],item['controller'],item['state_index'])]
            ki=item['candidate_index'];sp=menus[g['family']]['candidate_pool'][ki-1]
            c=make(scene_map[(g['type_id'],g['level'])]['scene'],ctl_map[g['controller']],g['state_index'],sp,purpose='v3_oracle_coverage')
            assert c['rollout_uid']==item['target_query_uid']
            frontier.append((g,c,ki))
        rows=requests([c for g,c,ki in frontier],pending)if frontier else{}
        labels=[]
        for g,c,ki in frontier:
            if c['rollout_uid']in rows:labels.append(consume(g,c,rows[c['rollout_uid']],ki))
            else:g['gap']='Infrastructure incomplete after retries; not a 64-candidate failure.'
        dump(HERE/f'{pending}_labels.json',labels)
        fi=int(pending.split('_')[1][1:]);round_number=int(pending.rsplit('_',1)[1][1:])
        progress[str(fi)]=round_number
        dump(HERE/'SEARCH_STATE.json',dict(groups=groups,baseline=baseline,active_family=fi,
             completed_batches=progress,pending_stage=None))
    for fi in ORDER:
        members=[g for g in groups if g['family']==fi]
        for batch in range(progress.get(str(fi),0),64):
            active=[g for g in members if not g['success']and not g['skipped']and not g['gap']and g['searched']<64]
            if not active:break
            stage=f'oracle_f{fi}_r{batch+1:02d}'
            frontier=[];labels=[];cache_consumed=[]
            with db.connect(True)as con:
                for g in active:
                    # Walk the frozen order until success, 64 outcomes, or the
                    # first genuinely absent result. Never inspect later eta.
                    while not g['success']and g['searched']<64:
                        ki=g['searched'];sp=menus[fi]['candidate_pool'][ki]
                        c=make(scene_map[(g['type_id'],g['level'])]['scene'],ctl_map[g['controller']],g['state_index'],sp,
                               purpose='v3_oracle_coverage')
                        r=cached(c,stage,con)
                        if r is None:frontier.append((g,c,ki+1));break
                        labels.append(consume(g,c,r,ki+1));cache_consumed.append(c)
            flush()
            dump(HERE/f'{stage}_cache_consumption.json',dict(exact_contexts=cache_consumed))
            dump(HERE/f'{stage}_frontier.json',[dict(family=fi,type_id=g['type_id'],level=g['level'],
                 controller=g['controller'],state_index=g['state_index'],candidate_index=ki,
                 target_query_uid=c['rollout_uid'])for g,c,ki in frontier])
            # Save cache-only progress before entering an expensive batch, so a
            # hard budget stop cannot hide reached historical successes.
            dump(HERE/'SEARCH_STATE.json',dict(groups=groups,baseline=baseline,active_family=fi,
                 completed_batches=progress,pending_stage=stage))
            write(groups,baseline)
            cases=[c for g,c,ki in frontier]
            rows=requests(cases,stage)if cases else{}
            if not cases:dump(HERE/f'{stage}_execution.json',dict(complete=True,new=0,cache_consumed=len(labels),missing=[]))
            for g,c,ki in frontier:
                uid=c['rollout_uid'];r=rows.get(uid)
                if r is None:
                    g['gap']='Infrastructure incomplete after retries; not a 64-candidate failure.'
                    continue
                labels.append(consume(g,c,r,ki))
            dump(HERE/f'{stage}_labels.json',labels)
            progress[str(fi)]=batch+1
            dump(HERE/'SEARCH_STATE.json',dict(groups=groups,baseline=baseline,active_family=fi,
                 completed_batches=progress,pending_stage=None))
            result=write(groups,baseline)
            status(FAMILY_NAMES[fi],sum(g['success']or g['searched']==64 or g['skipped']for g in members),len(members),
                   f'缺失候选批次 {batch+1}/64；当前找到成功 {sum(g["success"]for g in members)}/784；实际新运行 {used()}/90000。')
            print('COVERAGE',fi,batch+1,'success',sum(g['success']for g in members),'actual_new',used(),flush=True)
        dump(HERE/f'family_{fi}_complete.json',dict(family=fi,complete=all(g['success']or g['searched']==64 or g['skipped']for g in members)))
        if fi==4 and(HERE/'BASELINE_AFTER_FAMILY4.json').exists():
            from baseline_screen import main as paper_screen
            paper_screen()
    result=write(groups,baseline)
    paper_required=(HERE/'BASELINE_AFTER_FAMILY4.json').exists()
    paper_result=read(HERE/'paper_after_four_column/COMPLETE.json')if(HERE/'paper_after_four_column/COMPLETE.json').exists()else{}
    dump(HERE/'COMPLETE.json',dict(complete=result['complete']and(not paper_required or paper_result.get('complete',False)),
        correction_search_complete=result['complete'],actual_new_rollouts=used(),hard_limit=90000,
        family_order=ORDER,report=str(HERE/'REPORT.md'),no_dimension_selection=True,
        paper_methods_after_four_column=(HERE/'BASELINE_AFTER_FAMILY4.json').exists()))

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
