from core import *
from collections import Counter
prereg();r=read(HERE/'results.json');f=read(HERE/'FAILURE_RECORDS.json');unknown=read(HERE/'UNKNOWN_CASES.json');p=read(HERE/'main_protocol.json');lines=['## Additional delivery context','',
 '同一批TT评估state下，4个Flow参照仍有明显时间表收益（最佳时间表−最佳固定 +37.50pp，CI [19.14,61.33]），而MPC类为−1.82pp，势场类持平。因此，本轮跨类负结果不能仅归因于把FF修正位置换成TT，更符合这些nominal controller行为不同；结论限这两个实现类别与所测参数。势场F13已达到100%，其持平有天花板限制，不能据此证明所有势场controller都不存在阶段效应。','',
 '| class | F10 % | F13 % | F2 % | S200 % | S300 % | S500 % |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for k in ['M','P','Flow']:lines.append('| '+k+' | '+' | '.join(f'{100*x:.2f}' for x in r['groups'][k]['mean_rates'])+' |')
lines+=['','MPC和势场的冻结A3、A0均对全部6实例选择S500，故O3比较严格相等、CI为[0,0]，并非仅仅统计功效不足。类内O4在MPC上对6实例都选F13，改善1.82pp；这体现同类标签识别了共享较优臂，不能单凭该结果认定学到了细致的controller条件关系。势场O4在F13/S500间选择，两者均满分。P类为确定性nominal，Q2是两次真实执行，不把两个seed视作独立未来噪声的额外证据。','',
 '| class | known failure types | terminal step range | unknown | first error step range |','| --- | --- | --- | ---: | --- |']
for kind in ['M','P','Flow']:
 names=[n for n,v in p['base_profiles'].items() if v['payload']['controller_class']==kind];known=[x for n in names for x in f.get(n,[])];uu=[v for v in unknown if v['case']['base_controller_name'] in names];steps=[x['episode_length'] for x in known];err=[v['error']['step'] for v in uu];lines.append(f"| {kind} | {dict(Counter(v['outcome'] for v in known))} | {[min(steps),max(steps)] if steps else 'NA'} | {len(uu)} | {[min(err),max(err)] if err else 'NA'} |")
lines+=['','更宽的历史兼容性审计见HISTORICAL_TT_COMPATIBILITY_AUDIT.json：匹配checkpoint并检查结构化chain/placement及state/eta/seed，没有找到可直接用于这些gate案例的旧TT记录，故使用独立原TT入口重复一致性门。不能仅因新metadata UID不同便判旧标签失效。','',
 '实现修复前的controller/base protocol快照、原pilot manifest/preflight重建均保留，PREFLIGHT_V1_RECONSTRUCTION_AUDIT逐hash匹配原收据；当前运行使用V2冻结身份。状态区间修订为本轮已披露的预注册偏差。所有模型/探测冻结、修正后state去重和preflight均在主评估前完成。']
report=HERE/'REPORT.md';report.write_text(report.read_text().split('## Additional delivery context')[0].rstrip()+'\n\n'+'\n'.join(lines)+'\n')
r['bootstrap']=dict(repetitions=5000,base_seed=202610152721,group_order=['M','P','Flow','NEW'],derived_seed='base_seed + group_index; arm means use additional offset100',hypothesis_bounds='O1 compares lower CI endpoints under unknown assignments; O3 compares upper CI endpoints. Sampling-CI non-significance alone is not relabeled unknown uncertainty.');write(HERE/'results.json',r)
current=[]
for receipt in ['R5_FROZEN.receipt.json','CHOICES_FROZEN.receipt.json','CORRECTED_EVAL_STATES_FROZEN.receipt.json','CONTROLLER_IMPLEMENTATION_V2_FROZEN.receipt.json','main_PREFLIGHT.receipt.json','pilot_PREFLIGHT_V2.receipt.json']:
 rr=read(HERE/receipt)
 for path,digest in rr['files'].items():assert sha(path)==digest,(receipt,path)
 current.append(dict(receipt=receipt,utc=rr['utc'],files_checked=len(rr['files'])))
mapping={'controllers.py':HERE/'controllers_before_config_repair.py','base_protocol.json':HERE/'base_protocol_before_config_repair.json'}
for path,digest in read(HERE/'CONTROLLER_IMPLEMENTATION_FROZEN.receipt.json')['files'].items():assert sha(mapping.get(Path(path).name,Path(path)))==digest
assert all(read(HERE/'PREFLIGHT_V1_RECONSTRUCTION_AUDIT.json')['checks'].values());assert read(HERE/'UNIT_TESTS.json')['status']=='PASS' and read(HERE/'NATIVE_CONSTRUCTOR_TESTS.json')['status']=='PASS';assert read(HERE/'POSTFLIGHT.json')['status']=='PASS'
assert read(HERE/'HISTORICAL_TT_COMPATIBILITY_AUDIT.json')['structured_TT_older_same_state_eta_seed_records']==0
audit=read(HERE/'FINAL_AUDIT.json');audit.update(utc=utc(),results_sha256=sha(HERE/'results.json'),report_sha256=sha(report),frozen_lineage_checked=current,original_freeze_preserved_via_exact_snapshots=True,original_pilot_preflight_exact_reconstructed=True,unit_tests_final_code=True);write(HERE/'FINAL_AUDIT.json',audit)
print('Round27 final delivery and frozen lineage PASS')
