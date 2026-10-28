"""Read-only scientific summaries; never alter simulation or labels."""
from core import *
from collections import Counter
import subprocess,datetime

def longest(flags):
    best=run=0
    for x in flags:
        run=run+1 if x else 0;best=max(best,run)
    return best

def failure_type(result,scene):
    if not result.get('timeout'):return dict(category='数值异常或碰撞',outcome=result.get('outcome'),trace=result.get('trace'))
    with np.load(result['trace'])as trace:
        remaining=trace['route_remaining'];positions=trace['positions'];actions=trace['actions']
        history=json.loads(str(trace['admission_json']))if 'admission_json'in trace else []
        recorded_right_of_way=bool(history)
        dt=scene['config']['dt'];window=round(2/dt)
        moving=bool(len(remaining)>window and remaining[-window-1].sum()-remaining[-1].sum()>.05)
        gates=read(HERE/'PASSAGES.json')['by_scene'][scene['name']]['passages'];breaches=[];blocks=[]
        if not history and gates:
            # The old baseline has no admission state. Geometric two-way
            # occupancy is still measurable; exit-block attribution is not.
            history=[]
            for t,pos in enumerate(positions[:-1]):
                step=[]
                for gate in gates:
                    u=np.asarray(gate['axis']);n=np.array([-u[1],u[0]]);local=pos-gate['centre']
                    inside=(np.abs(local@u)<=gate['half_length']+scene['config']['agent_radius'])&(np.abs(local@n)<=gate['width']/2)&(remaining[t]>scene['config']['goal_tolerance'])
                    dirs=np.sign((trace['route_guides'][t]-pos)@u).astype(int)
                    step.append(dict(inside=inside.tolist(),robot_direction=dirs.tolist(),direction=0))
                history.append(step)
        for j,gate in enumerate(gates):
            mixed=[];blocked=[[]for _ in positions[0]]
            for t,step in enumerate(history):
                if j>=len(step):continue
                m=step[j];inside=np.asarray(m['inside']);dirs=np.asarray(m['robot_direction']);s=m['direction']
                mixed.append(bool(np.any(inside&(dirs==1))and np.any(inside&(dirs==-1))))
                slow=inside&(dirs==s)&(np.linalg.norm(actions[t],axis=1)<.02)if s else np.zeros(len(inside),bool)
                ex=np.asarray(gate['centre'])+s*gate['half_length']*np.asarray(gate['axis'])
                other=(dirs!=s)&(np.linalg.norm(positions[t]-ex,axis=1)<=2*scene['config']['agent_radius']+.05)
                for i in range(len(slow)):blocked[i].append(bool(slow[i]and np.any(other)))
            breaches.append(longest(mixed)*dt);blocks.append(max((longest(flags)*dt for flags in blocked),default=0))
        a=any(x>=2 for x in breaches);b=any(x>=2 for x in blocks)
        category='通行权失守'if a else'出口被堵'if b else'仍在推进的超时'if moving else'通道外拥堵或其他'
        return dict(category=category,flags=dict(right_of_way_breach=a,exit_blocked=b if recorded_right_of_way else None,progressing_timeout=moving),recorded_right_of_way=recorded_right_of_way,
                    max_bidirectional_seconds=max(breaches,default=0),max_exit_blocked_seconds=max(blocks,default=0),trace=result['trace'])

def write_report():
    family=read(HERE/'FAMILIES_THIS_ROUND.json')if(HERE/'FAMILIES_THIS_ROUND.json').exists()else dict(families=[])
    hard=read(HERE/'HARD_CASES.json')['cases']if(HERE/'HARD_CASES.json').exists()else[]
    groups=read(HERE/'ORACLE_STATE.json')['groups']if(HERE/'ORACLE_STATE.json').exists()else[]
    names={e['type_id']:e['title']+('，每侧'+str(e['num_agents']//2)+'台'if e['type_id']in(8,10,11)else'，'+str(e['num_agents'])+'台')for e in read(HERE/'REPRESENTATIVE17_FROZEN.json')['selected']}
    main=[g for g in groups if g['family']==0];mh=[g for g in main if g['hard']]
    cov=lambda rr,k:sum(g['success']and g['first_success']<=k for g in rr)
    oldpath=HERE/'inputs/final_v3/REPRESENTATIVE17_COMPARISON.json'
    old=read(oldpath)if oldpath.exists()else read(HERE/'inputs/opening/REPRESENTATIVE17_COMPARISON.json')
    reference=next(f for f in old['families']if f['index']==4)
    types=[]
    for t in sorted({g['type_id']for g in hard}):
        rr=[g for g in mh if g['identity'][0]==t];success=[g['first_success']for g in rr if g['success']]
        types.append(dict(type_id=t,name=names[t],cases=sum(g['type_id']==t for g in hard),main={str(k):cov(rr,k)for k in [16,32,64]},
            ablations=[cov([g for g in groups if g['family']==fi and g['identity'][0]==t],16)for fi in [1,2,3]],median=float(np.median(success))if success else None))
    stats=[]
    for fi in range(4):
        rr=[g for g in groups if g['family']==fi];success=[g['first_success']for g in rr if g['success']]
        stats.append(dict(family=fi,cases=len(rr),success=sum(g['success']for g in rr),mean_first_success=float(np.mean(success))if success else None,
            mean_completion_time=float(np.mean([g['solution']['result']['completion_time']for g in rr if g['success']]))if success else None,
            completed=sum(g['success']or g['searched']>=(64 if fi==0 and g['hard']else 16)for g in rr),
            unknown_queries=sum(q['result'].get('numerical_failure',0)for g in rr for q in g['queries'])))
    result=dict(hard_cases=len(hard),hard_main={str(k):cov(mh,k)for k in [16,32,64]},all_main_at16=cov(main,16),all_cases=272,
                four_column_at16=reference['coverage']['16'],by_type=types,families=stats,
                full_search_complete=bool(groups)and all(g['success']or g['searched']>=(64 if g['family']==0 and g['hard']else 16)for g in groups),
                short_horizon_policy='4x from executed 6x-first-success trajectories only; no search of skipped candidates.',
                scope='Finite candidate coverage; no family/dimension selection, no training, no paper-controller runs.')
    selected={(x['type_id'],x['level'])for x in read(HERE/'REPRESENTATIVE17_FROZEN.json')['selected']}
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']if(e['type_id'],e['level'])in selected}
    horizon=[]
    for g in groups:
        if not g['success']:continue
        sc=scenes[tuple(g['identity'][:2])];r=g['solution']['result']
        horizon.append(dict(family=g['family'],identity=g['identity'],hard=g['hard'],first_success=g['first_success'],success_4x=r['episode_length']<=sc['time_limits']['4']['max_steps'],success_6x=True))
    result['horizon_comparison']=horizon
    fixed=read(HERE/'FIXED_RESULTS.json')if(HERE/'FIXED_RESULTS.json').exists()else dict(methods={})
    fixed_table=[]
    for t,l in sorted(selected):
        row=dict(type_id=t,name=names[t])
        for method,d in fixed['methods'].items():
            a=[r for r in d['records']if r['identity'][0]==t]
            row[method]={str(start):dict(success=sum(bool(r['result']and r['result']['success'])for r in a if start<=r['identity'][3]<end),
                observed=sum(r['result']is not None for r in a if start<=r['identity'][3]<end),cases=16 if start==0 else 8)for start,end in [(0,8),(8,12)]}
        fixed_table.append(row)
    result['fixed_table']=fixed_table;dump(HERE/'COVERAGE.json',result)
    lines=[f"# 通道单向放行验证\n\n主族在当前冻结的{len(hard)}个困难案例上，Oracle@16 / @32 / @64 已验证覆盖为 {cov(mh,16)} / {cov(mh,32)} / {cov(mh,64)}；全部272案例上的Oracle@16当前为{cov(main,16)}/272，原四列为{reference['coverage']['16']}/272。未完成项不能视为失败。逐类型见下表。\n",
           '四列族为朝目标、互斥分离、目标右侧旋转、局部净空梯度；主族在其后加入等待方减速、等待方让出出口、通道内逆向者退出，并维护每个通道的通行权。它含状态与乘性减速，本轮不把它称为固定线性五维基，也不选择维度。\n',
           '名义控制器为集中式随机采样模型预测控制、梯度下降模型预测控制；沿用原两次安全投影、动力学、路线引导和时限。目标右侧旋转沿用历史命名，原字典正系数实际为逆时针，负系数为顺时针；未改变其实现。修正族和角色重分配定义见冻结输入说明。\n',
           '| 场景 | 困难案例数 | 主族16/32/64 | 只减速16 | 只放行16 | 角色重分配16 | 首次成功中位数 |\n|---|---:|---|---:|---:|---:|---:|']
    for x in types:lines.append(f"| {x['name']} | {x['cases']} | {x['main']['16']}/{x['main']['32']}/{x['main']['64']} | {' | '.join(map(str,x['ablations']))} | {x['median']} |")
    lines+=['\n固定参数：状态0–7每格16案例，状态8–11每格8案例；括号内为已观测数量。\n','| 场景 | 四列固定·0–7 | 放行固定·0–7 | 四列固定·8–11 | 放行固定·8–11 |\n|---|---|---|---|---|']
    for r in fixed_table:
        cells=[]
        for start,method in [(0,'four_column_fixed'),(0,'admission_fixed'),(8,'four_column_fixed'),(8,'admission_fixed')]:
            x=r.get(method,{}).get(str(start));cells.append(f"{x['success']}/{x['cases']}（观测{x['observed']}）"if x else'未运行')
        lines.append('| '+r['name']+' | '+' | '.join(cells)+' |')
    lines+=['\n每个案例按冻结候选顺序首次成功即停。主族并不包含四列全部原动作作为零放行候选，因此不预设覆盖单调改善；伤害案例将逐条报告。等待方减速作用于已累加的完整投影前输入，安全层仍可能改写动作，不是硬保持约束。\n',
            '旧两阶段补救结果只用于冻结困难子集。状态8–11仅用于固定参数检查；不能把开发集Oracle覆盖解释成部署成功率或连续空间最优。\n']
    for name in ['FAILURE_ANALYSIS.json','FIXED_PAIRED_INTERVALS.json','REGRESSIONS.json','ACCELERATION_CERTIFICATE.json']:
        if(HERE/name).exists():lines.append(f'\n[{name}]({name})\n')
    if(HERE/'LITERATURE_MAPPING.md').exists():lines.append('\n[列与文献约定对应](LITERATURE_MAPPING.md)\n')
    (HERE/'REPORT.md').write_text('\n'.join(lines).replace('None','—')+'\n')
    return result

def finalize():
    coverage=write_report();state=read(HERE/'ORACLE_STATE.json');fixed=read(HERE/'FIXED_RESULTS.json')
    scenes={(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']};failures=[]
    for g in state['groups']:
        if g['family']==0 and g['hard']and not g['success']and g['searched']==64:
            failures.append(dict(scope='hard_oracle_first_candidate',identity=g['identity'],**failure_type(g['queries'][0]['result'],scenes[tuple(g['identity'][:2])])))
    for method,m in fixed['methods'].items():
        for r in m['records']:
            if r['result']and not r['result']['success']:
                failures.append(dict(scope=method,identity=r['identity'],**failure_type(r['result'],scenes[tuple(r['identity'][:2])])))
    dump(HERE/'FAILURE_ANALYSIS.json',dict(records=failures,counts_by_scope={scope:dict(Counter(r['category']for r in failures if r['scope']==scope))for scope in sorted({r['scope']for r in failures})},classification_priority=['通行权失守','出口被堵','仍在推进的超时','通道外拥堵或其他'],independent_flags_also_reported=True))
    # Paired physical-state blocks keep both nominal controllers together.
    a={tuple(r['identity']):r['result']for r in fixed['methods'].get('four_column_fixed',{}).get('records',[])}
    b={tuple(r['identity']):r['result']for r in fixed['methods'].get('admission_fixed',{}).get('records',[])}
    intervals=[];rng=np.random.default_rng(2026102902);difficult={8,10,11,15,16}
    for start,end in [(0,8),(8,12)]:
        for name,predicate in [('全部',lambda t:True),('困难五类',lambda t:t in difficult),('其余类型',lambda t:t not in difficult)]:
            blocks={}
            for key,x in a.items():
                if not(start<=key[3]<end)or not predicate(key[0])or x is None or b.get(key)is None:continue
                block=(key[0],key[1],key[3]);blocks.setdefault(block,[]).append(int(b[key]['success'])-int(x['success']))
            vals=np.array([np.mean(v)for v in blocks.values()if len(v)==2])
            boot=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(axis=1)if len(vals)else[]
            intervals.append(dict(scope=name,states=[start,end-1],physical_blocks=len(vals),paired_difference=float(vals.mean())if len(vals)else None,ci95=np.quantile(boot,[.025,.975]).tolist()if len(vals)else None))
    dump(HERE/'FIXED_PAIRED_INTERVALS.json',dict(records=intervals,seed=2026102902,bootstrap_samples=10000,unit='physical scene/state, both controllers in same block'))
    old=read(HERE/'inputs/final_v3/SEARCH_STATE.json')['groups'];oldmap={(g['type_id'],g['level'],g['controller'],g['state_index']):g for g in old if g['family']==4}
    regressions=[dict(identity=g['identity'],new_queries=g['searched'],gap=g['gap'],old_first_success=oldmap[tuple(g['identity'])]['first_success'])for g in state['groups']if g['family']==0 and not g['hard']and oldmap[tuple(g['identity'])]['success']and not(g['success']and g['first_success']<=16)]
    dump(HERE/'REGRESSIONS.json',dict(cases=regressions,count=len(regressions),reference='four-column covered within64, new main not covered within16, as requested; unequal-budget comparator explicitly retained'))
    write_report()
    complete=coverage['full_search_complete']and all(m['complete']for m in fixed['methods'].values())
    dump(HERE/'COMPLETE.json',dict(complete=False,required_numeric_work_complete=complete,actual_new_attempts=read(HERE/'BUDGET.json')['actual_attempts'],hard_limit=15000,optional_sensitivity='pending',report=str(HERE/'REPORT.md')))
    (HERE/'STATUS.md').write_text('# 当前状态\n\n规定主线已完成。详见REPORT.md；可选附表和最终完整性核验见GAPS.md。\n')

