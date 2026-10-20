from common import *
import shutil

def main():
 assert read(HERE/'REPRODUCTION_GATE.json')['passed']
 assert read(HERE/'COPY_EQUIVALENCE.json')['passed']
 archive=HERE/'previous_source_boundary_stop_delivery';archive.mkdir(exist_ok=True)
 for n in ['REPORT.md','HANDOFF.md','results.json','GAPS.json','FINAL_DELIVERY.receipt.json','EXPERT_GATE.md','NAVIGATION_GATE.md','AXIS_TABLE.md','AXIS_TABLE.csv','SCALING_CURVE.csv','INTEGRITY_FINAL_COMPARISON.json']:
  if (HERE/n).exists():shutil.copyfile(HERE/n,archive/n)
 text='''用户授权从2026-10-17源码边界停止处恢复。提示词第零节新增决定明确：旧提示只列轮次而缺少实际目录名属于提示词错误；误读只作边界偏差，不算实验失败。SOURCE_BOUNDARY_FAILURE.json和原INTEGRITY_BEGIN.json原样保留，禁止目录中内容未用于实验，未写入，之后不再读取。

恢复审计按diagnostics/decentralized_*与diagnostics/teacher_student_*前缀剪枝；decentralized_pipeline_v1为共同协议只读例外。禁止目录和并行目录只记录os.stat元数据。采用INTEGRITY_RESUME_BEGIN.json为本次终止审计基线。已过16条复现、原几何划分、联合证据和专家门登记均复用，不重做。对4096条专家门官方再预检和直接身份检查，只提交仍缺失的条目。专家设置、任务、门槛不变。

采集示范前实现39维训练和三轮专家重标注流程并通过单元测试，封存代码及固定训练配置。门不通过则不采集该族示范，继续另一试点；两试点未达标时按原提示停止扩展并交付诊断。'''
 (HERE/'PREREGISTRATION_RESUME.md').write_text(text+'\n')
 seal('PREREGISTRATION_RESUME.receipt.json',[HERE/'PREREGISTRATION_RESUME.md',ROOT/'docs/prompt_round40_multiscene_navigation_competence_20261017.md',HERE/'audit_resume.py',HERE/'INTEGRITY_RESUME_BEGIN.json'],reason='Explicit user recovery authorization; prompt clarified directory prefixes; prior failure retained',no_old_records_reexecuted=True)
 pf=planner.preflight(HERE/'expert_gate_PLANNED_ROLLOUTS.json');write(HERE/'expert_gate_CACHE_PREFLIGHT_RESUME.json',pf)
 rows=read(HERE/'expert_gate_CASES.json')
 from shared_rollout_db.src.rollout_db import connect
 with connect(True) as db:
  missing=[c for c in rows if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None]
 write(HERE/'expert_gate_MISSING_CASES_RESUME.json',missing)
 assert missing==read(HERE/'expert_gate_MISSING_CASES.json')
 assert all(sha(p)==v for p,v in read(HERE/'EXPERT_GATE_IMPLEMENTATION.receipt.json')['files'].items())
 seal('expert_gate_PREFLIGHT_RESUME.receipt.json',[HERE/'expert_gate_CACHE_PREFLIGHT_RESUME.json',HERE/'expert_gate_MISSING_CASES_RESUME.json'],before_submission=True,planned=len(rows),genuinely_missing=len(missing),unknown_rerun=False)
 print('recovery preregistered; missing',len(missing))
if __name__=='__main__':main()
