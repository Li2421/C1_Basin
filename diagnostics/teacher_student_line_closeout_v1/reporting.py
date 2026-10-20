from core import *
import csv
precision()
def fmt(r,percent=False,digits=2):
 if r is None or r.get('estimate') is None:return '未定义'
 scale=100 if percent else 1;suffix='%' if percent else '';v=f"{r['estimate']*scale:.{digits}f}{suffix}";ci=r.get('CI95')
 return v+(f" [{ci[0]*scale:.{digits}f}, {ci[1]*scale:.{digits}f}]{suffix}" if ci else '')
def table(name,headers,rows,records=None,intro=''):
 (HERE/(name+'.md')).write_text(intro+'\n\n'+'| '+' | '.join(headers)+' |\n|'+'---|'*len(headers)+'\n'+'\n'.join('| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows)+'\n')
 recs=records or [dict(zip(headers,row)) for row in rows]
 if recs:
  with (HERE/(name+'.csv')).open('w') as f:
   keys=list(dict.fromkeys(k for r in recs for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(recs)
def main():
 a1=read(HERE/'analysis1_teacher_ceiling.json');a2=read(HERE/'analysis2_nominal_failure.json');a3=read(HERE/'analysis3_safety_stalls.json');a4=read(HERE/'analysis4_correction.json');a5=read(HERE/'analysis5_zero_controller.json');a6=read(HERE/'analysis6_relabeling_shared_noise.json');paper=read(HERE/'analysis7_paper_cells.json');secondary=read(HERE/'SECONDARY_ANALYSES.json')
 rows=[[f"冻结的联合MACFlow教师（{r['teacher']}）",r['frame'],*[f"{100*r[k]:.2f}%" for k in ['teacher_nominal','teacher_safety','V0','V1','V3']]] for r in a1['per_teacher']]
 table('ANALYSIS1_TEACHER_CEILING',['教师全名','坐标系','只跑教师','教师只加分布式安全层','原学生只跑','闭环重标注学生只跑','共享随机种子学生只跑'],rows,a1['per_teacher'],'每个教师/学生对应32个state×2seed，n=64；全部DEVELOPMENT。逐教师差值及state簇CI见SECONDARY_ANALYSES.json。')
 rows=[]
 for g,r in a2['groups'].items():
  types=secondary['exclusive_collision_types'][g];rows.append([FULL[g],fmt(r['outcomes']['success']['lower'],True),fmt(r['outcomes']['collision']['lower'],True),fmt(r['outcomes']['timeout']['lower'],True),fmt(r['mixed_failure'],True),fmt(r['mixed_success'],True),fmt(r['four_robots_same'],True),fmt(r['head_on_among_agent_collision'],True),'; '.join(f'{k}={v}' for k,v in types.items())])
 table('ANALYSIS2_NOMINAL_FAILURE',['控制器全名','成功率','碰撞率','超时率','失败局不一致','成功局不一致','四机器人全部同向','迎面/机器人间碰撞','互斥碰撞类型计数'],rows,intro='每组n=896，14controller×32state×2seed；来源round30/30b/30c/30f。CI先controller再state簇重抽5000次，DEVELOPMENT。专家32/32全队一致。agent=机器人间，center=中心障碍，outer=外环；multi单列。')
 rows=[]
 for g,r in a3['groups'].items():
  rows.append([FULL[g],r['timeout_episodes'],fmt(r['last_progress_seconds']),fmt(r['final_unfinished']),fmt(r['has_low_speed_stall'],True),fmt(r['stall_robot_seconds']),fmt(r['moving_no_progress_robot_seconds']),fmt(r['has_geometric_opposite_pair'],True),f"{r['location_comparable']}/{r['collision_to_timeout']}",fmt(r['near_original_collision_050'],True)])
 table('ANALYSIS3_SAFETY_STALLS',['控制器全名','已知超时局','最后显著进展秒','未到达机器人数','有40步低速停滞','低速停滞机器人秒','仍运动但无进展机器人秒','有几何相向反向对','位置可比/碰撞转超时','最终距原碰撞≤0.5m'],rows,intro='仅只加分布式安全层的已知超时局；指标为条件均值/比例，CI为配对设计的controller/state分簇重抽。位置比较只使用原碰撞涉及且最终未到达的同一机器人。移动无进展为滚动2秒窗口端点累计，不等于全局死锁。全部DEVELOPMENT。')
 rows=[]
 for g,r in a4['paired_transitions'].items():
  for a,q in r.items():
   early=a4['early_progress'][g][a];base=a4['early_progress'][g]['safety baseline'];rows.append([FULL[g],ARM_FULL[a],fmt(q['mixed_or_stationary_to_consistent'],True),int(q['mixed_or_stationary_to_consistent']['denominator']),fmt(q['consistent_rate_difference'],True),fmt(early['100']['progress']),fmt(base['100']['progress']),fmt(early['100']['four_same'],True),fmt(base['100']['four_same'],True),fmt(q['rescue_of_baseline_timeout'],True)])
 table('ANALYSIS4_CORRECTION',['控制器全名','修正臂全名','不一致/静止转一致','该转换分母','最终一致率配对差','前100步进展m','安全层前100步进展m','修正100步四机器人同向','安全层100步四机器人同向','超时救回率'],rows,intro='同controller/state/seed配对；转换分母只包含安全层下不一致或静止的局，不能将其转换率解释为全部失败的机制解释。全部DEVELOPMENT。')
 rows=[]
 for a,r in a5['zero'].items():rows.append([ARM_FULL[a],fmt(r['outcomes']['success']['lower'],True),fmt(r['consistent_failures'],True),fmt(r['mixed_failures'],True),fmt(r['four_same_failures'],True),fmt(r['moving_no_progress_failures']),fmt(r['stalls_failures']),fmt(r['final_unfinished_failures'])])
 table('ANALYSIS5_ZERO_CONTROLLER',['零动作控制器修正臂','成功率','失败局一致','失败局不一致','失败局四机器人同向','失败局移动无进展机器人秒','失败局低速停滞机器人秒','失败局未到达数'],rows,intro='来源round30d阶段A分布式安全层；每臂64条记录、32个独立state。两个seed的positions/actions逐位相同，CI只以32个state为独立簇。全部DEVELOPMENT。')
 rows=[]
 for k,q in a6['paired'].items():
  label='闭环重标注学生减原学生（V1−V0）' if k.startswith('V1') else '闭环重标注加共享随机种子学生减闭环重标注学生（V3−V1）';rows.append([label,fmt(q['success']['lower'],True),fmt(q['mixed_rate_difference'],True),fmt(q['collision_rate_difference'],True),fmt(q['timeout_rate_difference'],True),fmt(q['moving_no_progress_robot_seconds_difference']),fmt(q['moving_no_progress_difference_both_timeout'])])
 table('ANALYSIS6_RELABELING_SHARED_NOISE',['方法全名及比较','成功率差','不一致率差','碰撞率差','超时率差','移动无进展机器人秒差','双方均超时子集中的时长差'],rows,intro='14个教师身份配对，32state×2seed；全部DEVELOPMENT。全局时长差受碰撞提前截断的暴露时长影响；双方均超时的子集提供相同时长敏感性，仍为选择后的描述性比较。')
 # Wide requested paper table, with each numeric cell carrying its own provenance and sample count.
 methods=ARMS+['critic','oracle'];headers=['控制器全名']+[ARM_FULL.get(a,'冻结的通用critic选择（A3）' if a=='critic' else '四修正臂内按控制器事后最优（oracle）') for a in methods];rows=[];records=[]
 for g in ['expert','TC','TD','V0','V1','V3','Z']:
  cells=[]
  for a in methods:
   r=paper[g].get('expert' if g=='expert' and a=='nominal only' else a,dict(status='MISSING',reason='No compatible historical cell'))
   if r.get('status')=='MISSING':cells.append('缺');records.append(dict(controller=FULL[g],group=g,method=headers[methods.index(a)+1],arm=a,status='MISSING',reason=r.get('reason')));continue
   sl=r['success']['lower'];su=r['success']['upper'];co=r['collision']['lower'];to=r['timeout']['lower'];source=', '.join(r['source_rounds']);u=r['success']['unknown'];cell=f"成功 {fmt(sl,True)}；上界 {su['estimate']:.2%}；碰撞 {co['estimate']:.2%}；超时 {to['estimate']:.2%}；unknown={u}；n={r['n']}；{source}；DEVELOPMENT";cells.append(cell);records.append(dict(controller=FULL[g],group=g,method=headers[methods.index(a)+1],arm=a,status='AVAILABLE',success_lower=sl['estimate'],success_upper=su['estimate'],CI_low=sl['CI95'][0],CI_high=sl['CI95'][1],CI_unknown_low=r['success']['CI_unknown_envelope'][0],CI_unknown_high=r['success']['CI_unknown_envelope'][1],collision_lower=co['estimate'],collision_upper=r['collision']['upper']['estimate'],timeout_lower=to['estimate'],timeout_upper=r['timeout']['upper']['estimate'],unknown=u,n=r['n'],independent_states=r['independent_states'],controllers=r['controllers'],source_rounds=source,evidence_level='DEVELOPMENT'))
  rows.append([FULL[g],*cells])
 table('PAPER_TABLE',headers,rows,records,'全部DEVELOPMENT；每格分别注明来源轮次、样本数及unknown。集中式专家的“只跑”格是其原生特权规划参照；其余臂未运行，均为缺。两行联合教师的只跑格复用同一组896条记录，不是新增样本。零动作控制器每臂64条记录只有32个独立state。冻结critic仅复用历史冻结的通用critic选择（A3）；事后最优按每个controller的总体成功率选四修正臂之一，bootstrap内重新选臂，不逐局挑选。oracle碰撞/超时率按全样本选定臂描述；各项完整CI保留在analysis7_paper_cells.json。')
 # Reviewed plotting denominator: cumulative success uses all requested episodes.
 curves=read(HERE/'PLOT_DATA.json')
 for g,arms in curves.items():
  for a,q in arms.items():
   n=q['n_known']+q['unknown'];lo=np.asarray(q['success_cumulative_fraction'])*q['n_known']/n;q['success_cumulative_fraction']=lo.tolist();q['success_cumulative_upper']=(lo+q['unknown']/n).tolist();q['n_requested']=n;q['progress_denominator']='known episodes only; unknown not extrapolated';q['success_denominator']='all requested episodes'
 write(HERE/'PLOT_DATA_FINAL.json',curves);write(HERE/'TABLES_COMPLETE.json',dict(utc=utc(),tables=7,all_DEVELOPMENT=True,DB_writes=0,precision=precision()));print('tables complete',utc(),flush=True)
if __name__=='__main__':main()
