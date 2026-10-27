"""Outcome-blind state selection and historical checks before new experiments."""
from experiment import *
from source_inputs import load,verify
from flow_scene import validate
import collections
import numpy as np

def historical_baseline(scene,qualification):
    qualified={r['controller'] for r in qualification if r['qualified']}
    groups=collections.defaultdict(list);types={};paths=[]
    for c in load(V1/f'safety_{scene}_CASES.json'):
        ctl=c['controller_payload']['nominal']
        if ctl['controller'] not in qualified:continue
        p=V1/f'safety_{scene}/raw'/f"{c['rollout_uid']}.json"
        r=load(p);groups[ctl['controller']].append(r);types[ctl['controller']]=ctl['type_code']
        paths.append(dict(path=str(p),sha256=sha(p),rollout_uid=c['rollout_uid']))
    rates={k:float(np.mean([r['success'] for r in v])) for k,v in groups.items()}
    pertype=collections.defaultdict(list)
    for k,v in rates.items():pertype[types[k]].append(v)
    allrows=[r for rows in groups.values() for r in rows]
    return dict(qualified=len(qualified),episodes=len(allrows),success=sum(r['success'] for r in allrows),
                pooled_episode_rate=float(np.mean([r['success'] for r in allrows])),
                type_equal_rate=float(np.mean([np.mean(v) for v in pertype.values()])),
                controller_rates=rates,unknown=sum(r['numerical_failure'] for r in allrows),sources=paths)

def main():
    assert not (HERE/'PREREGISTRATION.receipt.json').exists()
    allstates=load(V1/'FLOW_STATES_UNCHECKED.json');mainsets={};splits={};geometry=[]
    for scene in NAMES:
        old=sorted([s for s in allstates if s['scene']==scene],key=lambda s:(s['N'],s['index']))
        if scene=='lit_crowded_random':
            valid={}
            for s in old:
                row=validate(s);geometry.append(row)
                if row['eligible']:valid.setdefault(s['N'],[]).append(s)
            mainsets[scene]=[s for n in (2,4,6,8,10,12,14) for s in valid[n][:16]]
            train=[s for n in (8,14) for s in valid[n][16:32]]
            evaluation=[s for s in mainsets[scene] if s['N'] in (8,14)]
            assert len(mainsets[scene])==112 and len(train)==len(evaluation)==32
        elif scene=='lit_circle_five':
            mainsets[scene]=old[:16];train=old[16:48];evaluation=old[:16]
        else:
            mainsets[scene]=old
            original=load(V1/f'LEARNING_SPLIT_{scene}.json')
            train=original['training'];evaluation=original['evaluation']
            assert len(old)==64
        for s in mainsets[scene]:
            if scene!='lit_crowded_random':assert validate(s)['eligible']
        assert not ({(s['N'],s['index']) for s in train}&{(s['N'],s['index']) for s in evaluation})
        splits[scene]=dict(training=train,evaluation=evaluation)
        write(HERE/f'FLOW_STATES_{scene}.json',mainsets[scene])
        write(HERE/f'LEARNING_STATE_SPLIT_{scene}.json',splits[scene])
    write(HERE/'GEOMETRY_VALIDATION_lit_crowded_random.json',geometry)
    capstates=mainsets['lit_crowded_random']+splits['lit_crowded_random']['training']
    write(HERE/'CAPABILITY_STATES_lit_crowded_random.json',capstates)
    assert len(capstates)==144 and sum(s['N'] for s in capstates)==1248
    qualifications={}
    qfiles={'lit_narrow_passage':'capability_v2_lit_narrow_passage_QUALIFICATION.json',
            'lit_doorway':'capability_lit_doorway_QUALIFICATION.json',
            'lit_circle_five':'capability_lit_circle_five_QUALIFICATION.json'}
    for scene,name in qfiles.items():qualifications[scene]=load(V1/name)
    history={s:historical_baseline(s,load(V1/(qfiles.get(s,'capability_lit_crowded_random_QUALIFICATION.json'))))
             for s in ('lit_narrow_passage','lit_crowded_random','lit_doorway')}
    write(HERE/'HISTORICAL_BASELINE_RECHECK.json',history)
    gates=[]
    for scene,name in qfiles.items():
        stage='capability_v2_'+scene if scene=='lit_narrow_passage' else 'capability_'+scene
        picked={}
        for c in load(V1/f'{stage}_CASES.json'):
            s=c['state'];ctl=c['controller_payload']['nominal']
            if c['seed']!=0 or s['index']!=0 or s['agent']!=0:continue
            if ctl['type_code'] in picked:continue
            path=V1/stage/'raw'/f"{c['rollout_uid']}.json";r=load(path)
            assert not r['numerical_failure']
            picked[ctl['type_code']]=dict(kind='capability_reuse',scene=scene,source=str(path),source_sha256=sha(path),
                                         trace=r['trace_path'],trace_sha256=r['trace_sha256'])
        assert len(picked)==8,(scene,picked.keys());gates.extend(picked.values())
    for filename,kind in (('CONTROLLER_INTERIOR_REPLAY.json','central_historical'),('TILED_PARENT_REPLAY.json','planner_storage')):
        rows=load(V1/filename);assert len(rows)==16
        for row in rows:
            verify(row['source']);verify(row['trace'])
            gates.append(dict(kind=kind,source=row['source'],source_sha256=row['source_sha256'],trace=row['trace'],trace_sha256=row['trace_sha256']))
    write(HERE/'HISTORICAL_GATE_CASES.json',gates)
    circle_rows=[load(p) for p in sorted((V1/'safety_lit_circle_five/raw').glob('*.json'))]
    retained_main=sum(r['case']['state']['index']<16 for r in circle_rows)
    design=dict(utc=utc(),evidence='DEVELOPMENT',main_state_counts={s:len(v) for s,v in mainsets.items()},
        training_state_counts={s:len(v['training']) for s,v in splits.items()},
        evaluation_state_counts={s:len(v['evaluation']) for s,v in splits.items()},
        capability_crowded_states=144,capability_crowded_agent_states=1248,capability_crowded_requests=119808,
        distributed_baseline_blackbox_requests=24576,distributed_baseline_whitebox_requests=512,
        central_crowded_baseline_requests=10752,central_circle_baseline_requests=1536,
        central_circle_completed_v1_main16_candidates=retained_main,
        capability_replay_count=24,central_replay_count=16,planner_replay_count=16,
        seeds=[0,1],label_seed=0,bootstrap_replicates=5000,
        qualified_reuse={s:sum(r['qualified'] for r in rr) for s,rr in qualifications.items()},
        fixed_request_formulas=dict(distributed='38496 + 672*Q_crowded',central_new='4320 + 672*Q_crowded'),
        learning_label_requests_if_all_informative=24576,learning_eval_requests_if_all_informative=10752,
        stage2_requests='51328 + 896*Q_crowded',
        new_semantics='Distributed safety/correction; no-wall crowded geometry; paper dynamics each have new execution fingerprints. Unchanged central circle/narrow/door retain exact v1 identities.',
        circle_v1_stop_boundary='Discarded v1 requests/claims are never resumed. The explicit v2 main16 experiment creates its own independently preflighted requests/queues; only completed v1 evidence is reused.',
        rng='PRNGKey(2026102004), fold SHA256(state_uid)[:8], seed, then physical step; same physical state_uid retained across safety layers when geometry is unchanged.',
        controller_semantics='One nominal actor call per physical step, preserving stateful controller warm starts; only safety/correction execution is decentralized.',
        optional_dynamics=dict(double_integrator_l0=1.,double_integrator_l1=2.,circle_l0=6.,circle_l1=5.,
            acceleration_bound=1.5,circle_acceleration_bound=None,velocity_barrier_gamma=1.,
            doorway_lookahead=.025,doorway_inflated_radius=.125,doorway_v_bounds=[-.26,.26],
            doorway_omega_bounds=[-3.8,3.8],doorway_discrete_gamma=.1,initial_headings=0.,
            guarantee='Positive auxiliary Euler factors require lambda*dt<=1 (lambda=1 for non-circle); this alone is not a sampled-hold or infeasible-QP safety guarantee. Actual constraint violations and swept collisions remain evidence; no post-step safety correction.'))
    write(HERE/'STUDY_DESIGN.json',design)
    write(HERE/'RESOURCE_AUTHORIZATION.json',dict(authority='Latest explicit user instruction in this conversation',
        total_cpu_shares=24,total_gpu_shards=5,local_cpu_reserve=1,check_all_user_squeue_before_every_submission=True,
        overrides_prompt_original_limits=[18,2],no_v1_jobs_remaining=True))
    lines=['# 历史证据与复用边界','',
        'v1已在2026-10-20T15:59:55Z冻结，全部作业结束。论文版2707条记录及PAPER_REPRODUCTION系列全部只读沿用，不重跑；v1严格开工审计缺少两份非Python配置整文件哈希的限制一并继承，本轮开工已另作完整输入哈希。','',
        '|场景|合格实例|旧基线局数|按局成功率|类型等权成功率|','|---|---|---|---|---|']
    for s,r in history.items():lines.append(f"|{NAMES[s]}|{r['qualified']}|{r['episodes']}|{100*r['pooled_episode_rate']:.6f}%|{100*r['type_equal_rate']:.6f}%|")
    lines+=['','上述按局率与类型等权率使用不同权重，不能混用。逐实例数和全部来源sha256在HISTORICAL_BASELINE_RECHECK.json；旧4a候选为0见v1已封存SAFETY_BASELINE.csv，新4a′将在本轮预注册后重算。',
        f'圆形围场旧682条完成raw保留，其中{retained_main}条位于本轮前16个主state；这是待精确preflight核验的候选，不把682都算作本轮复用。8918条旧未落盘请求不记unknown、不恢复；本轮新范围有独立身份清单和队列。','',
        '第28轮可读例外：decentralized_pipeline_v1/REPORT.md、PREREGISTRATION.md、pipeline.py与HANDOFF.md均已复核。原圆环14个Flow策略、32state、Q2、12544条主评估中，shared/F10为2.68%、shared/F13为31.25%、shared/S500为75.56%；central/F10下界12.39%、central/S500下界93.08%。该线未设正式deadlock判据，低速/timeout不是死锁，策略仍共享全队观测与公共RNG。其几何、controller、state及动作定义与本轮不一致，仅复用局部投影与修正基实现，不复用标签。',
        '第28轮以后的其他decentralized_/teacher_student_/multiscene_/baseline1_目录仍按前缀剪枝，仅stat；不能宣称检查了其内部结论。其他近24小时可能并行写入的目录也只stat。所需8类48实例、η菜单与参数已在v1封存副本中取得，不读取这些并行目录。','',
        '窄通道、门口与圆形能力门复用必须先通过每场景8条N1逐位回放。拥挤随机去墙，N1约束及controller墙代价都变了，必须新跑能力门。分布式多机投影与逐机器人修正是新执行语义；旧集中式标签不因此变成分布式标签。仅更换critic输入不会使相同执行语义的标签失效。']
    (HERE/'HISTORY_CHECK.md').write_text('\n\n'.join(lines)+'\n')
    write(HERE/'WORK_PROGRESS.json',dict(utc=utc(),phase='protocol_design_before_gates',new_research_rollouts=0,
        audit='INTEGRITY_START complete; input manifest frozen',next='Seal protocol, run unit and byte-equivalence gates, then register/preflight crowded capability',
        v1_read_only=True,resources='24 CPU shares / 5 GPU shards total; reserve one local CPU'))
    print('Design',design['main_state_counts'],'new crowded capability',design['capability_crowded_requests'],flush=True)

if __name__=='__main__':main()
