"""Close the mandatory stop honestly; never mark the requested experiment complete."""
from audit import HERE,V2,sha,write
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess

def main():
    end=json.loads((HERE/'INTEGRITY_END.json').read_text())
    stop=json.loads((HERE/'READONLY_BOUNDARY_STOP.json').read_text())
    impact=json.loads((HERE/'READONLY_BOUNDARY_IMPACT.json').read_text())
    pilot=json.loads((HERE/'lit_circle_five_initial_PILOT_RESULTS.json').read_text())
    covered_pass=end['passed']
    end.update(covered_files_unchanged=not end['changed_files'],covered_git_matches=end['git_matches'],
               protected_cache_violation=True,protected_cache_impact_sha256=sha(HERE/'READONLY_BOUNDARY_IMPACT.json'),
               missing_opening_import_hashes=json.loads((HERE/'SUPPLEMENTAL_IMPORT_AUDIT.json').read_text())['files'],
               passed=False,verdict='FAILED: protected v2 runtime cache was written; whole experiment stopped')
    write(HERE/'INTEGRITY_END.json',end)
    journal_dir=HERE.parents[1]/'shared_rollout_db/journals/exp_baseline_literature_scenes_stage2_collision_fix_v1'
    journals=list(journal_dir.glob('*.jsonl')) if journal_dir.exists() else []
    write(HERE/'DB_STOP_AUDIT.json',dict(utc=datetime.now(timezone.utc).isoformat(),official_journals=len(journals),
          journal_files={str(p):sha(p) for p in journals},merger_run=False,
          disposition='Official default-path journals preserved unmerged after source-protection stop; no extra SQLite copy or direct rollout inserts',
          accepted_scientific_cohort=False))
    report=f'''完整性审计**未通过，本轮已整体停止，任务未完成**。已覆盖的{len(end['protected'])}个文件内容比较与两个仓库Git状态/diff比较为{'一致' if covered_pass else '存在差异'}；但复制的原生CBS库内置v2缓存绝对路径，实际写入封版v2的纯查询缓存。仅用os.stat核查发现{impact['modified_since_opening_count']}个缓存文件在开工后被写入。证据：READONLY_BOUNDARY_STOP.json、READONLY_BOUNDARY_IMPACT.json、INTEGRITY_END.json。另有一个间接导入模块缺少最初开工哈希，已如实披露，未倒补审计。未删除、回滚或改写v2文件。\n\n修复后的零碰撞目标**尚未达成**。圆形初始试跑完成{pilot['completed']}/360局，其中{pilot['collisions']}局碰撞、{pilot['unknown']}局unknown；因整体停止，剩余7局不是unknown，不继续运行。唯一离散回退完成迎面门和10000随机快照门，gamma=.3、解析区间最小中心距满足4.0001，但回退试跑尚未启动，不能报告为碰撞修复成功。\n\n窄通道与二维拥挤随机的随机快照单元门分别发现256和2个“原约束有解但本步碰撞”的反例，尚未启动研究rollout。支持半空间推导需要预测构型安全和已有区间可存活条件；这些任意速度快照不等同于从零初速度研究轨迹可达的状态。门口第二阶段及第一阶段仍按原身份复用；本轮三场景全量研究、5000次配对bootstrap和修复增益结论均为N/A。旧原始结果已重算，不伪造缺失结论。\n\n本轮实际完成353条新试跑轨迹，raw、trace、诊断和{len(journals)}个官方默认路径journal均保留，因源码保护停机尚未运行merger，不应并入已接受主研究证据。本轮自有Slurm作业均已取消或结束，资源守护已停止；09:00时本轮没有继续占用的worker，不存在待释放的本轮CPU份额。未来恢复时仍须遵守新加坡2026-10-23 09:00后18总CPU份额（17worker+1管理核）的用户授权。\n\n根因是复用预编译共享库前未检查其内置文件写路径。不能只复制Python文件和二进制就声称输出边界已隔离。后续安全实现应先在本轮目录重建具有本轮缓存路径的原生库，或使用已证等价且完全不写历史目录的后端；须先证明计算等价并核验全部写路径，再依据用户对本次强制停机的处理决定恢复。该修改目前**未执行**，不在停机后擅自继续实验。\n'''
    (HERE/'REPORT.md').write_text(report)
    (HERE/'STATUS.md').write_text(f'''# STOPPED_SOURCE_PROTECTION\n\n整体停止，任务未完成；没有COMPLETE.json。原因是原生CBS库写入v2只读缓存。所有本轮Slurm作业与守护已停止，未启动回退和全量。\n\n圆形试跑353/360，碰撞2、unknown0，其余7局未完成；结果隔离保留。窄通道/拥挤随机未通过随机快照单元门，无研究rollout。回退单元门通过但无回退试跑。\n\n{len(journals)}个默认路径journal保留未合并；开工后被写缓存文件数{impact['modified_since_opening_count']}。源码、物理和成功判据没有主动修改；完整性审计总体失败，不能用覆盖文件未变冒充全面通过。详见REPORT、GAPS、HANDOFF和INTEGRITY_END。\n''')
    with (HERE/'GAPS.md').open('a') as f:f.write(f'''\n## 强制停机（优先于此前条目）\n\n原生CBS预编译库写入只读v2查询缓存，整体停止；发现{impact['modified_since_opening_count']}个开工后修改的缓存文件，未做清理或恢复。不得继续执行旧入口、回退或全量。\n\n圆形初始试跑最终353/360（2碰撞），剩余7局因强制停机未完成；离散回退只完成单元门，未运行试跑。此前“正在执行”文字仅为历史进度，不代表当前仍在执行。353份新轨迹及官方journal隔离保存、未merger；67712全量请求未完成。\n\n完整性覆盖额外缺口：exact_projection_retry.py的间接导入缺最初开工哈希，补充记录不能倒补成首次使用前证据。\n''')
    with (HERE/'HANDOFF.md').open('a') as f:f.write('''\n## 强制停止覆盖上述运行说明\n\n当前STOPPED_SOURCE_PROTECTION，**不要执行本文件前面列出的入口**。编译CBS库内置v2缓存根目录；复制后的库依然写v2。源码保护违规已触发整体停机。dispatch和当前worker入口已加停止哨兵，所有作业与资源guard已停。保留353条raw/trace及未合并journal；不清理、不回滚v2缓存。详情REPORT与READONLY_BOUNDARY_IMPACT。必须先处理边界事件和恢复条件，不能接手续跑。没有COMPLETE.json。\n''')
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('''\n8. 检查原生缓存实现时发现预编译CBS库内置v2缓存根路径；立即取消4个剩余自有Slurm worker并停止资源guard。仅os.stat核查影响，不删除、恢复或继续写v2。依据用户源码保护规则整体停下，不启动已规划的回退，不运行merger，不写COMPLETE。\n''')
    write(HERE/'STOPPED.json',dict(utc=datetime.now(timezone.utc).isoformat(),status='STOPPED_SOURCE_PROTECTION',
          user_task_complete=False,scientific_success=False,active_slurm_jobs=[],new_raw=pilot['completed'],
          official_journals_unmerged=len(journals),audit_passed=False,
          reports={n:sha(HERE/n) for n in ('REPORT.md','STATUS.md','GAPS.md','DYNAMICS_STAGE.md','HANDOFF.md','INTEGRITY_END.json')}))
    print('Closed as STOPPED, not complete;',len(journals),'journals preserved unmerged')
if __name__=='__main__':main()
