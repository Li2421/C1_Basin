from core import *
from guard import guard
from audit_utils import source_inventory
G=guard()
def main():
 r=read(HERE/'results.json');audit=read(HERE/'INTEGRITY_COMPARISON.json');e=r['endpoints']
 def decision(key):
  x=e.get(key,{})
  return '证据不足' if 'passes' not in x else '通过' if x['passes'] else '未通过'
 first=f"源码完整性审计{'通过' if audit['protected_content_pass'] else '未通过'}，Basin_C1 共享数据库与 journal 路径断言通过。闭环重标注学生（各自噪声；V1）相对原学生（离线蒸馏、各自噪声；V0）的能力提升检验{decision('V1_competence')}；闭环重标注加共享随机种子学生（V3）相对闭环重标注学生（各自噪声；V1）的额外提升检验{decision('V3_competence')}。通过两道门之后，前500步使用第13组修正参数、之后使用第10组修正参数的时间表（S500）相对只加安全层的检验，在两类新学生上分别{decision('V1_deadlock_one')}、{decision('V3_deadlock_one')}；相对公平调参右手规则（RHR-fair）分别{decision('V1_deadlock_two')}、{decision('V3_deadlock_two')}。失败构成的实测数量和比例见下表；timeout仅称停滞诊断，不等同已证实死锁。"
 text=first+'\n\n资源规则有一次已记录的例外：初次复现作业22197提交时总CPU为19，随后立即取消；Slurm accounting未启用，持续时间未知。之后由配额检查限制提交。此例外不隐去，不宣称全程资源合规。\n\n'
 text+='旋转测试采用场景与global世界坐标噪声共同旋转的向量等变定义；固定同一随机键时global噪声不随场景旋转，不能宣称逐样本旋转不变。此区别在预注册中披露。\n\n'
 text+='全部新学生最终版本固定为第三轮第20,000步。原学生和教师为DEVELOPMENT参照，新学生为CONFIRMATORY；主比较只使用双方通过两个门的同教师配对交集，排除项单独保留。unknown按上下界报告，不重跑。\n\n'
 text+=(HERE/'PAPER_TABLE.md').read_text()+'\n'+(HERE/'FAILURE_COMPOSITION.md').read_text()+'\n配对终点与5000次controller→state分簇bootstrap：\n\n```json\n'+json.dumps(e,ensure_ascii=False,indent=2)+'\n```\n\n冻结critic的遗憾及与一律时间表的配对差见results.json的critic字段；各臂与对应联合教师的差距及缺口见gaps字段。教师缺少精确兼容的臂时不补造结果。\n'
 (HERE/'REPORT.md').write_text(text)
 handoff='''工作目录为本文件所在目录；Python为/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python。任务来源：docs/prompt_round30f_student_competence_dagger_shared_noise_20261016.md，只读。只允许写本目录和官方DB追加流程。

请先读REPORT.md、results.json、RUN_STATUS.json、PREREGISTRATION.md、RESOURCE_EXCEPTION.json、INTEGRITY_COMPARISON.json。COPY_PROVENANCE.json和COPY_DIFF.patch列出复制来源；IMPORTED_SOURCE_AUDIT.json列出实际导入源码hash。所有checkpoint、闭环采集npz、标签、逐轮统计、训练history和门预测均保留。不要重新训练冻结学生，不要重跑unknown。

coordinator.py负责资源检查、复现、训练DAG、全局冻结、pilot、R5、盲选择、主评估、单进程官方merger、统计与交付。没有后台并发时方可启动；先检查COORDINATOR_PID和ps。所有Slurm提交走quota.submit，用户总18CPU/2GPU shard，提交前先读squeue。资源任务失败写GAP并继续独立部分，不自动重试；需要接续时读coordinator_state.json及GAPS.json。不得以改旧源码、包或环境绕过失败。

所有新评估按design→preflight封存→仅missing→默认journal→官方单merger→postflight。合并前检查ps和squeue。复现为任务指定的16条历史记录逐位重放，单列，不写DB重复标签。闭环采集只用于训练，不进入评估统计。
'''
 (HERE/'HANDOFF.md').write_text(handoff);source_inventory('imports_delivery.json');seal('DELIVERY.receipt.json',[HERE/'REPORT.md',HERE/'HANDOFF.md',HERE/'results.json',HERE/'PAPER_TABLE.md',HERE/'FAILURE_COMPOSITION.md',HERE/'INTEGRITY_COMPARISON.json'],all_required_stages_attempted=True)
if __name__=='__main__':main()
