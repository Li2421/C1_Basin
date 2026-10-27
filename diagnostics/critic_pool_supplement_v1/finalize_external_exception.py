from common import *
from audit_pool import integrity
assert read(HERE/'DB_POSTFLIGHT.json')['passed'] and read(HERE/'HISTORICAL_ROWS_END.receipt.json')['passed']
first=(HERE/'REPORT.md').read_text().split('\n\n')[1]
integrity();post=read(HERE/'DB_POSTFLIGHT.json');assert post['passed']and post['all_requests_present']
with(HERE/'REPORT.md').open('a')as f:f.write(f"\n新正式主标签{post['new_main_requests']}条，含能力门共{post['new_requests']}条新请求；默认共享数据库身份、结果与轨迹逐项核对通过，无未合并journal。已有训练行、旧源码和全部冻结文件审计通过。\n")
(HERE/'STATUS.md').write_text('# 当前状态\n\n'+utc()+'：COMPLETE。标注、14版本抽取、critic训练、冻结、验证、统计及最终审计全部完成。\n\n'+first+'\n')
files=['REPORT.md','HISTORY_CHECK.md','SCENES.json','GEOMETRY_REPORTS.json','CAPABILITY_TABLE.json','LABEL_TABLES.json','TRAINING_POOL.json','FROZEN.receipt.json','TRAINING_FROZEN.receipt.json','STATISTICS.json','TEST_DECISION.json','STATUS.md','GAPS.md','ITERATION_LOG.md','DB_POSTFLIGHT.json','HISTORICAL_ROWS_END.receipt.json','INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARE.json','PARALLEL_METADATA_END.json','PER_SCENE_RESULTS.md','SUPPLEMENT_EFFECTS.md','NEW_SCENES.md','CANDIDATE_RATES.md','PER_CELL_RESULTS.md','TRAINING_POOL_CV.md','LEARNING_CURVES.md','TOOLING_COPY.receipt.json','VALIDATION_LABEL_ACCESS.receipt.json','TRAINING_PLAN.json']
write(HERE/'DELIVERY_CODE_SHA256.json',{str(p):sha(p)for p in HERE.glob('*.py')});files.append('DELIVERY_CODE_SHA256.json')
write(HERE/'COMPLETE.json',dict(utc=utc(),status='COMPLETE',decisions=read(HERE/'STATISTICS.json')['decisions'],new_main_requests=post['new_main_requests'],new_requests=post['new_requests'],source_and_database_audits_passed=True,reports={n:sha(HERE/n)for n in files}));write(HERE/'RUN_STATE.json',dict(status='COMPLETE',active_jobs=[],report=str(HERE/'REPORT.md')));print(first,flush=True)
