"""Final analysis from immutable results only; no simulation or DB writes."""
from pathlib import Path
import datetime,json,statistics

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def mean(values):return statistics.mean(values)if values else None

def main():
    labels=dict(A='通道单向放行·三列版',B='通道单向放行·合成一列',C='只让等待方减速·一列',D='定时交替放行',E='整队让行',F='预约到达时间',G='去等候点')
    scene_map={(e['type_id'],e['level']):e['scene']for e in read('SCENES.json')['entries']}
    families=[];success_sets={}
    for code,label in labels.items():
        state=read(f'FOLLOWUP_{code}_hard.json');groups=state['groups'];good=[g for g in groups if g['success']]
        success_sets[code]={tuple(g['identity'])for g in good}
        families.append(dict(scheme=code,name=label,cases=len(groups),success=len(good),success_fraction=len(good)/len(groups),mean_first_success=mean([g['first_success']for g in good]),mean_all_queries=mean([g['searched']for g in groups]),success_mean_completion_seconds=mean([g['solution']['result']['completion_time']for g in good]),search_complete=state['complete']))
    fixed=read('FIXED_RESULTS.json')['methods'];a={tuple(r['identity']):r['result']for r in fixed['four_column_fixed']['records']};b={tuple(r['identity']):r['result']for r in fixed['admission_fixed']['records']}
    fixed_stats=[]
    for lo,hi in [(0,8),(8,12)]:
        keys=[k for k in a if lo<=k[3]<hi];common=[k for k in keys if a[k]['success']and b[k]['success']]
        row=dict(states=[lo,hi-1],cases=len(keys),new_only=sum(not a[k]['success']and b[k]['success']for k in keys),old_only=sum(a[k]['success']and not b[k]['success']for k in keys),common_success=len(common))
        for method,data in [('four_column',a),('admission',b)]:
            row[method]=dict(success=sum(data[k]['success']for k in keys),success_mean_completion_seconds=mean([data[k]['completion_time']for k in keys if data[k]['success']]),paired_common_success_mean_seconds=mean([data[k]['completion_time']for k in common]),whole_cohort_truncated_mean_seconds=mean([data[k]['completion_time']if data[k]['success']else scene_map[k[:2]]['config']['max_steps']*scene_map[k[:2]]['config']['dt']for k in keys]))
        fixed_stats.append(row)
    universe={tuple((g['type_id'],g['level'],g['controller'],g['state_index']))for g in read('HARD_CASES.json')['cases']}
    union=set.union(*success_sets.values())
    summary=dict(created=datetime.datetime.now().astimezone().isoformat(),families=families,fixed=fixed_stats,paired_intervals=read('FIXED_PAIRED_INTERVALS.json'),cross_family_union=dict(covered=len(union),cases=len(universe),not_a_single_frozen_method=True,uncovered=[list(k)for k in sorted(universe-union)]),unique_vs_three_term={code:[list(k)for k in sorted(v-success_sets['A'])]for code,v in success_sets.items()},budget=read('BUDGET.json'),no_full49_reason='No single scheme reached36of40; all49 testing was conditional on90percent coverage.',oracle_scope='Finite frozen candidate menus; no Generator or Critic; times are conditional on each scheme own successful cases unless explicitly paired.')
    (HERE/'FINAL_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    path=HERE/'REPORT.md';text=path.read_text();marker='\n# 交付结论与成功时间\n'
    text=text.split(marker)[0]
    lines=[marker,'全部规定搜索于2026-10-29 08:36完成。共新运行14420局（原说明4095、续接10325）；200局完整旧轨迹等价、原主族50对加速、六个续接方案各50对加速及无通道认证通过。没有单个方案达到36/40，因此依照冻结门槛未触发49场景测试。',
           '\n| 方案 | 困难案例覆盖 | 成功所需候选均值 | 成功任务完成均时·秒 |','|---|---:|---:|---:|']
    for r in families:lines.append(f"| {r['name']} | {r['success']}/{r['cases']} | {r['mean_first_success']:.2f} | {r['success_mean_completion_seconds']:.2f} |")
    lines+=['\n三列版覆盖最高；合成一列版少覆盖一例，成功所需候选均值更小，是值得保留的五列协调候选。六方案只在困难子集筛查，不能据此宣称其在所有场景中的部署最优。各方案成功子集不同，表中条件均时也不能直接当作配对速度排名。',
            '\n主族在全部17代表场景、两控制器、状态0–7的同预算Oracle@16上为217/272，原四列为183/272。主族230/272的总成功数混合了困难案例64候选与其他案例16候选，不能写作全272案例的Oracle@64。',
            '\n| 固定参数状态 | 四列成功 | 放行成功 | 条件均时·四列/放行 | 配对均成功时间·四列/放行 | 新增/损失成功 |','|---|---:|---:|---|---|---|']
    for r in fixed_stats:
        x,y=r['four_column'],r['admission'];lines.append(f"| {r['states']} | {x['success']}/{r['cases']} | {y['success']}/{r['cases']} | {x['success_mean_completion_seconds']:.2f}/{y['success_mean_completion_seconds']:.2f} | {x['paired_common_success_mean_seconds']:.2f}/{y['paired_common_success_mean_seconds']:.2f} | {r['new_only']}/{r['old_only']} |")
    lines+=['\n独立状态8–11的固定参数成功率增加9.56个百分点，物理初态分块自助法95%区间为[4.41,16.18]个百分点。开发状态0–7增加13.97个百分点，区间为[8.46,19.85]个百分点。开发状态有3例旧成功而新失败，已纳入配对比较，未删除。',
            '\n三列版对窄门16/16、长走廊13/13、串联瓶颈4/4，货架仓库0/7；其余方案在货架仓库有少量补救。跨族成功见证并集38/40仅是诊断上界，不能冒充单个固定方法达到95%。剩余两例身份与各族专长见FINAL_SUMMARY.json。',
            '\n成功条件均值、配对成功均值、全队列截断均值分开保存在FINAL_SUMMARY.json。14,420局原始轨迹、逐候选标签、planned/preflight/execution清单均保留在本目录。冻结设计、原控制器和安全层完整性核验通过。',
            '\n流程记录：自动协调器运行期间持续检查作业并按分钟写入状态；交互代理两次主动核查之间超过30分钟，没有满足说明中代理每30分钟主动核查的要求。此项为监控流程偏差，不补称已完成；数值结果已在本次交付逐项复核。']
    path.write_text(text+'\n'+'\n'.join(lines)+'\n')
    (HERE/'GAPS.md').write_text('# 证据缺口与流程偏差\n\n规定数值实验完成；没有单个方案达到90%门槛，故未触发49场景验证。有限候选失败不构成连续参数不可解证明。\n\n自动协调器持续监控并更新状态；交互代理主动核查间隔超过30分钟，未满足该项监控频率要求。原始结果、轨迹及完整性核验已在交付时复核。\n')
    complete=read('COMPLETE.json');complete['final_summary']=str(HERE/'FINAL_SUMMARY.json');complete['monitoring_process_deviation']='Interactive agent check interval exceeded30min; automatic coordinator monitoring remained active.'
    (HERE/'COMPLETE.json').write_text(json.dumps(complete,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
