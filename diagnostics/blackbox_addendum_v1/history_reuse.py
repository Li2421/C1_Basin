from common import *
from registry import connect
from audit_utils import source_inventory
OLD=D/'axis4a_addendum_v1'
def main():
    prereg();assert not (HERE/'HISTORICAL_REUSE.receipt.json').exists();same={f:sha(HERE/f)==sha(OLD/f)==sha(R35/f) for f in ['runtime.py','factor_runtime.py','prefix.py','progress.py']};assert all(same.values());assert read(OLD/'CONFIG_UNIT_TEST.json')['passed'];assert read(OLD/'A1_STATE_DEDUP.json')['accepted']==16
    pilot=read(OLD/'A1_STATES.json');assert [s['draw_seed'] for s in pilot]==list(range(202610354000,202610354016));assert all(s['evaluation_config']==dict(source_config,max_steps=700,obstacle_radius=.9) for s in pilot for source_config in [read(HERE/'HISTORICAL_INVENTORY.json')['config']])
    proofs={};allproof=[]
    for label,casefile,indexfile in [('A1','A1_CASES.json','A_RAW_INDEX.json'),('A2','A2_CASES.json','A_RAW_INDEX.json'),('B','B_CASES.json','B_RAW_INDEX.json')]:
        idx={q['rollout_uid']:q for q in read(OLD/indexfile)};cases=read(OLD/casefile);rs=[]
        with connect(True) as db:
            for c in cases:
                ref=idx[c['rollout_uid']];assert sha(ref['raw_path'])==ref['raw_sha256'];raw=read(ref['raw_path']);assert sha(raw['trace_path'])==raw['trace_sha256'];row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None and row['compatibility_quality']=='EXACT_REUSE' and not row['conflict_quarantined'];assert all(row[k]==raw[k] for k in ['success','collision','timeout','numerical_failure','episode_length']);rs.append(dict(case=c,raw_path=ref['raw_path'],raw_sha256=ref['raw_sha256'],trace_path=raw['trace_path'],trace_sha256=raw['trace_sha256'],DB_row_sha256=hashlib.sha256(json.dumps(dict(row),sort_keys=True).encode()).hexdigest()))
        proofs[label]=dict(records=len(rs),source_cases_sha256=sha(OLD/casefile),proof_file=str(HERE/f'REUSED_{label}_EVIDENCE.json'));write(HERE/f'REUSED_{label}_EVIDENCE.json',rs)
    assert proofs['A1']['records']==448 and proofs['A2']['records']==1792 and proofs['B']['records']==2688;assert read(OLD/'PSTAR_DECISION.receipt.json')['pstar']==read(HERE/'DESIGN_CONFIG.json')['pstar']
    text='''# 历史证据复用说明（不改变实验参数）
在几何新仿真前的全DB物理三元组检索中，发现已完成的 diagnostics/axis4a_addendum_v1。它已执行相同种子、配置、动作和随机流的生成器阳性对照448条、跨半径修正臂1792条、以及用户同一候选1门槛的按进度切换2688条。本轮逐条检查原始结果、轨迹哈希、官方DB身份与标签一致性；四个数值执行源码与round35逐字相同，配置和复现门已有证据，本轮另行32条复现亦通过。

依AGENTS.md历史优先规则，复用这些已执行的实验。阳性对照的DB物理重复来自该对照本身，属于精确缓存命中；不把它误作“新候选碰撞”而顺延种子、重复回答已回答的问题。保留原去重发生时的快照和收据，不伪造当前新增state。跨半径只补只加安全投影和只执行controller动作的缺口；补齐旧能力门和新阳性对照的动作基线以完成失败分类。几何全量条件评估仍严格由原预注册规则决定，不改变门槛。

按进度切换部分仍在本轮几何部分完成之后独立重算699条中位数并写本轮决定收据，再核验/引用旧标签；没有新的按进度切换rollout。既有证据保留来源的等级和时间，不将复用冒充本轮全新标签。其余新比较与机器人数量试验的设计完全不变。此说明在任何新几何主要结果统计前封存，不根据本轮新标签改变参数。
'''
    (HERE/'HISTORICAL_REUSE_AMENDMENT.md').write_text(text);write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),source_directory=str(OLD),numeric_source_exact=same,reused=proofs,new_gaps=dict(A='Missing safety and nominal geometry arms/failure diagnostics only',B='No new rollout needed; independent calibration receipt and offline reanalysis required',D='No matching original-placement continuing-right-hand grid/eval evidence found; new semantics require labels',E='No matching N-independent distributed-shooting Gap controller profiles; transplant and timing/gates required'),read_only_F_C_complete=True));seal('HISTORICAL_REUSE.receipt.json',[HERE/'HISTORY_AUDIT.json',HERE/'HISTORICAL_REUSE_AMENDMENT.md']+[HERE/f'REUSED_{k}_EVIDENCE.json' for k in ['A1','A2','B']],only_evidence_origin_amended=True,no_parameter_changes=True,new_rollouts_avoided=4928);source_inventory('imports_history_reuse.json')
if __name__=='__main__':main()
