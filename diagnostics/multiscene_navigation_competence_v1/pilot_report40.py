"""Final failed-pilot delivery from completed native expert and learned-policy evidence."""
from common import *
from collect39 import POLICIES,checkpoint
from deliver_failed_expert_pilot import protected_status,NAMES
from pilot_diagnostics40 import analyze
import csv

def audit():
    begin=read(HERE/'INTEGRITY_RESUME_BEGIN.json');end=read(HERE/'INTEGRITY_RESUME_END.json');peers=set(begin['parallel_metadata_only'])|set(end['parallel_metadata_only'])
    changes=[];parallel=[]
    for p in sorted(set(begin['sources'])|set(end['sources'])):
        a=begin['sources'].get(p);b=end['sources'].get(p)
        if a==b:continue
        row=dict(path=p,before=a,after=b)
        try:is_peer=Path(p).relative_to(D).parts[0] in peers
        except ValueError:is_peer=False
        (parallel if is_peer else changes).append(row)
    passed=not changes and protected_status(begin['Basin']['status'],peers)==protected_status(end['Basin']['status'],peers) and begin['Basin']['head']==end['Basin']['head'] and begin['MAC']==end['MAC'] and begin['Basin']['diff_sha256']==end['Basin']['diff_sha256']
    result=dict(passed=passed,baseline='INTEGRITY_RESUME_BEGIN.json',end='INTEGRITY_RESUME_END.json',protected_source_changes=changes,parallel_source_changes=parallel,prohibited_metadata_begin=begin['prohibited_metadata_only'],prohibited_metadata_end=end['prohibited_metadata_only'],parallel_metadata_begin=begin['parallel_metadata_only'],parallel_metadata_end=end['parallel_metadata_only'],prior_boundary_deviation='User explicitly authorized recovery; original failure and initial audit retained unchanged')
    write(HERE/'INTEGRITY_FINAL_COMPARISON.json',result);return passed

def main():
    assert (HERE/'PILOT_DESCRIPTIVE_EXECUTION_COMPLETE.json').exists()
    experts=read(HERE/'EXPERT_GATE_RESULTS.json')['families'];families={};gaps=[];diagnostics={}
    for f,name in NAMES.items():
        gate=read(HERE/f'NAVIGATION_CAPABILITY_{f}.json') if (HERE/f'NAVIGATION_CAPABILITY_{f}.json').exists() else None
        trained=sum((checkpoint(f,k,s,3).parent/'COMPLETE.receipt.json').exists() for k,s in POLICIES)
        families[f]=dict(name=name,trained_instances=trained,navigation_gate=gate,passed_instances=sum(r['passed'] for r in gate['instances']) if gate else 0,expert_gate=experts.get(f))
        if f in 'BF':
            if gate:diagnostics[f]=analyze(f)
            if not experts[f]['passed']:reason='专家门未通过，不能采集示范或训练。'
            elif gate is None:reason='固定训练阶段受阻，详见保留的训练错误；未替换、未调参。'
            elif not gate['family_passed']:reason='最终第三轮策略通过导航能力门的实例少于两个。'
            else:continue
            gaps.append(dict(item=name,reason=reason,condition='后续研究需另立预注册，获得通过专家门并至少两个实例通过导航门的固定导航设计；本轮不调参或重跑。'))
        else:gaps.append(dict(item=name,reason='两族试点未同时通过，未进入全量。',condition='新预注册试点中两个试点族均至少两个实例通过导航门后，才允许扩展。'))
    pilot_passed=all(families[f]['navigation_gate'] and families[f]['navigation_gate']['family_passed'] for f in 'BF')
    assert not pilot_passed,'Pilot passed: continue full stage, do not deliver a failed-pilot report.'
    integrity=audit()
    first=('恢复后完整性审计通过。' if integrity else '恢复后完整性审计未通过，差异见审计记录。')+'试点判定不通过，按要求停止扩展。'+''.join(f"{NAMES[f]}（族 {f}）训练完成{families[f]['trained_instances']}个实例，通过导航能力门{families[f]['passed_instances']}个实例；" for f in 'BF')+'其余五族未进入全量。修正选择、规模曲线与随机障碍场锁定测试未运行。'
    expert_lines=[];navigation_lines=[]
    for f in 'BF':
        e=experts[f];c=e['counts'];expert_lines.append(f"{NAMES[f]}：专家成功{c['success']}/{e['episodes']}，撞墙{c['collision']}，超时{c['timeout']}，未知{c['unknown']}，专家门{'通过' if e['passed'] else '未通过'}。最低墙体扫掠余量分位数（米）：{json.dumps(e['minimum_wall_clearance_quantiles'],ensure_ascii=False)}。偏离原几何路径超过0.5米的回合比例{e['route_deviation_episode_rate']:.4%}、状态比例{e['route_deviation_state_rate']:.4%}。失败终点图：expert_diagnostics/{f}_expert_failure_positions.svg。")
        gate=families[f]['navigation_gate']
        if gate:
            for r in gate['instances']:
                kind='确定性行为克隆策略' if r['kind']=='deterministic' else '混合密度行为克隆策略'
                navigation_lines.append(f"{NAMES[f]}，{kind}，种子{r['seed']}：新留出几何成功{r['success']}/16，撞墙{r['collision']}，超时{r['timeout']}，未知{r['unknown']}，{'通过' if r['passed'] else '未通过'}。最终检查点：{r['checkpoint']}。")
        else:navigation_lines.append(f'{NAMES[f]}：导航能力门未运行，不能将四个未训练实例写为四次失败。')
    old=['上一轮专家碰撞率：串联瓶颈走廊257/640，房间与门134/640；上一轮合法留出几何最佳实例分别为0/16和1/16。本轮专家任务及几何数量不同，只作描述性对照。']
    for f in diagnostics:
        d=read(HERE/f'PILOT_POLICY_DIAGNOSTICS_{f}.json')
        for kind,seed in POLICIES:
            tag=f'{kind}_{seed}';rounds=d['training'].get(tag,{})
            for r in ['0','3']:
                if r in rounds:
                    x=rounds[r];navigation_lines.append(f"{NAMES[f]}，{'确定性行为克隆策略' if kind=='deterministic' else '混合密度行为克隆策略'}，种子{seed}，第{r}轮训练几何：成功{x['success']}/{x['episodes']}，撞墙{x['collision']}，超时{x['timeout']}，未知{x['unknown']}；偏离路线回合比例{x['route_deviation_episode_rate']:.4%}，状态比例{x['route_deviation_state_rate']:.4%}。失败位置图：{x['failure_plot']}。")
            if d['old_evaluation_descriptive']:
                x=d['old_evaluation_descriptive'][tag];old.append(f"{NAMES[f]}，{'确定性行为克隆策略' if kind=='deterministic' else '混合密度行为克隆策略'}，种子{seed}：本轮最终策略在上一轮原评估几何上描述性成功{x['success']}/16，撞墙{x['collision']}，超时{x['timeout']}，未知{x['unknown']}；这些结果不参与导航门。")
    (HERE/'EXPERT_GATE.md').write_text('\n\n'.join(expert_lines)+'\n')
    (HERE/'NAVIGATION_GATE.md').write_text('\n\n'.join(navigation_lines+old)+'\n')
    gaps.extend([dict(item='跨场景修正选择、规模曲线与随机障碍场锁定测试',reason='试点未通过，按阶段顺序停止；没有打开修正评估标签，没有遗憾或偏好结论。',condition='试点通过并扩展后，至少四个前六族各至少两个实例通过导航门；所有目标族选择先冻结，随机障碍场锁定测试在前六族分析冻结后仅评估一次。')])
    write(HERE/'GAPS.json',gaps)
    stages=[]
    for p in sorted(HERE.glob('*_POSTFLIGHT.json')):
        stage=p.name[:-len('_POSTFLIGHT.json')];raw=list((HERE/stage/'raw').glob('*.json'))
        stages.append(dict(stage=stage,executed=len(raw),postflight=str(p),unknown_preserved=read(p).get('unknown_preserved',[])))
    write(HERE/'DB_STAGE_INDEX.json',dict(stages=stages,official_workflow=True,unexecuted_expert_profiles=['expert_gate','expert_gate_v2'],no_unknown_reruns=True))
    training_status={}
    for f in NAMES:
        instances={}
        for kind,seed in POLICIES:
            rounds={}
            for r in range(4):
                cp=checkpoint(f,kind,seed,r);manifest=HERE/'datasets39'/f'{f}_{kind}_{seed}_round_{r}.json';summary=cp.parent/'SUMMARY.json'
                rounds[str(r)]=dict(status='COMPLETE' if (cp.parent/'COMPLETE.receipt.json').exists() else 'NOT_COMPLETE' if manifest.exists() else 'NOT_RUN',checkpoint=str(cp) if (cp.parent/'COMPLETE.receipt.json').exists() else None,data=read(manifest)['transitions'] if manifest.exists() else None,training_summary=read(summary) if summary.exists() else None)
            instances[f'{kind}_{seed}']=rounds
        training_status[f]=instances
    write(HERE/'TRAINING_STAGE_STATUS.json',dict(families=training_status,current_freeze=read(HERE/'TRAINING_PIPELINE39_CURRENT.json'),resource_failure_and_recovery=read(HERE/'TRAINING_OOM_RECOVERY_COMPLETE.json') if (HERE/'TRAINING_OOM_RECOVERY_COMPLETE.json').exists() else None))
    write(HERE/'results.json',dict(status='COMPLETE_WITH_FAILED_PILOT',integrity_passed=integrity,pilot_passed=False,families=families,gaps=gaps,correction_labels_opened=0,locked_test_evaluations=0,scaling_regret=None))
    methods=['仅执行控制器','只加安全投影','全程修正参数十','全程修正参数十三','全程修正参数十五','修正参数十三至十按比例切换','修正参数十五至十按比例切换','按进度切换','通用持续右手规则','零样本选择模型','第十六轮冻结的控制器级选择模型（A3）','第十六轮冻结的不看控制器的评价模型（A0）','探测估计后规则选择（G_E）','最佳固定参数交叉拟合参照','完整菜单样本内最佳参照','源训练场景平均最佳臂','少量标签选择','完整菜单交叉拟合最优参照','专家']
    header=['子项']+methods+['原因','证据等级'];rows=[[n]+['缺口：未评估']*len(methods)+['试点未通过','未产生修正终点证据'] for n in list(NAMES.values())+['前六族合并']]
    with (HERE/'AXIS_TABLE.csv').open('w') as out:
        w=csv.writer(out);w.writerow(header);w.writerows(rows)
    (HERE/'AXIS_TABLE.md').write_text('| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(r)+' |\n' for r in rows)+'\n导航专家与联合可解证据单独报告，不充作预算匹配的修正性能。\n')
    with (HERE/'SCALING_CURVE.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['训练族数','平均遗憾','状态']);w.writerows([[m,'NA','试点未通过，未运行'] for m in [0,1,3,5]])
    paragraphs=[first,'本轮修复导航示范、加入路线观测，并执行固定的三轮专家重标注。网络与超参数未根据导航留出结果调整；第 3 轮固定为最终策略，不替换未通过的实例。'+ '两族各64训练几何和8个全新导航留出几何，训练、开发、停止与检查点选择均不使用导航留出几何。',*expert_lines,'逐实例导航门、上一轮描述性对照、第零轮与第三轮失败位置图和撞墙、超时、路线偏离比例见NAVIGATION_GATE.md及PILOT_POLICY_DIAGNOSTICS_*.json。各轮数据量与训练几何成功率见TRAINING_ROUND_COUNTS_*.csv；所有最终检查点及开发集选择收据保留于policies39/。没有进入训练的族对应数据标为未运行。','路线偏离只描述距原Geometry.path超过0.5米的情况；避墙路线本身允许绕远，偏离不是额外失败判据。超时不称死锁，未知保留，不重跑。死锁修复仍需多机器人成功率相对只加安全投影的提升，本轮未产生这一证据。','恢复按用户补充决定执行。SOURCE_BOUNDARY_FAILURE.json及旧INTEGRITY_BEGIN.json保留，禁止目录按前缀剪枝，禁止与并行目录仅os.stat元数据；结束审计与INTEGRITY_RESUME_BEGIN.json比较。已过十六条复现、几何与联合证明复用，没有重做。全部新增源码与输出在本轮目录，官方共享数据库追加为唯一例外；没有Git写操作或包安装。','数据库流程为设计追加登记、官方预检、真正缺失执行、默认journal、单进程官方合并、复检，详见DB_STAGE_INDEX.json。资源按用户新增授权本轮最多六个处理器分片，图形处理器分片最多两个；每次提交前检查全用户队列。专家两排队任务依据实际内存峰值降低申请后六个处理器全部运行，执行代码不变。','两族未同时各有两个实例通过，因此按提示词停止扩展。其余五族、修正动作菜单、多机器人能力门、探测、选择模型、少量标签、规模曲线和随机障碍场锁定测试均未进入。不对时间表或持续右手规则作偏好判断。']
    if (HERE/'TRAINING_OOM_RECOVERY_COMPLETE.json').exists():paragraphs.append('第一轮训练曾因4GiB主机内存申请不足在编译时被Slurm终止；原日志、提交和未完成目录均保留。仅提高主机内存申请至8GiB后补完未完成训练，成功旧检查点和全部rollout数据复用，模型、优化器、样本、种子、固定开发集及选择规则未变。详见TRAINING_OOM_RECOVERY_COMPLETE.json和PREREGISTRATION_GPU_HOST_MEMORY8G.receipt.json。')
    (HERE/'REPORT.md').write_text('\n\n'.join(paragraphs)+'\n')
    (HERE/'HANDOFF.md').write_text(first+'\n\n任务提示词：docs/prompt_round40_multiscene_navigation_competence_20261017.md。先读REPORT.md、EXPERT_GATE.md、NAVIGATION_GATE.md、GAPS.json、INTEGRITY_FINAL_COMPARISON.json与DB_STAGE_INDEX.json。原源码边界停止交付在previous_source_boundary_stop_delivery/，用户已明确授权恢复，原失败记录与旧审计保留；当前为试点完成且未通过，停止扩展。\n\n写入边界为diagnostics/multiscene_navigation_competence_v1/，官方共享数据库追加为唯一例外。禁止目录按diagnostics/decentralized_*及diagnostics/teacher_student_*前缀剪枝，仅协议指定的decentralized_pipeline_v1允许只读；禁止与并行目录只记录os.stat元数据。不得Git写操作，不安装包。Python为/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python，使用-B及本轮目录缓存。\n\n训练入口见TRAINING_PIPELINE39_CURRENT.json指向的当前冻结版。旧35维脚本只作来源参照，不执行。训练第零轮专家示范和三轮专家重标注、最终第三轮、固定开发集选择已在任何数据前冻结；批量单机器人独立执行与参考逐位一致，物理与终止不变。导航能力门回合在PILOT_NAVIGATION_STATES_FROZEN.json，第3轮最终权重每实例仅评估一次，不替换、不重跑或调参。各轮数据与实际轨迹、最终权重及收据均保留。\n\n已过复现、几何、联合证据不重做。unknown保留，不视为已知失败；超时不称死锁。不得将导航成功或联合几何证明充作修正方法死锁改善。规模曲线与锁定测试尚无结果。若未来改变设置，必须另立研究预注册；本轮不自动调参或进入全量。\n')
    write(HERE/'ACTIVE_STAGE.json',dict(utc=utc(),status='COMPLETE_WITH_FAILED_PILOT'))
    names=['REPORT.md','HANDOFF.md','EXPERT_GATE.md','NAVIGATION_GATE.md','GAPS.json','results.json','AXIS_TABLE.md','AXIS_TABLE.csv','SCALING_CURVE.csv','DB_STAGE_INDEX.json','TRAINING_STAGE_STATUS.json','INTEGRITY_FINAL_COMPARISON.json']
    write(HERE/'FINAL_DELIVERY.receipt.json',dict(utc=utc(),status='COMPLETE_WITH_FAILED_PILOT',files={str(HERE/n):sha(HERE/n) for n in names}))
    print(first)

if __name__=='__main__':main()
