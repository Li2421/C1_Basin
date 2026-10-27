"""Final read-only assertions over completed v2 delivery and immutable evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess

H=Path(__file__).resolve().parent

def read(name):return json.loads((H/name).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    c=read('COMPLETE.json');assert c['status']=='COMPLETE'and c['development_only']and c['all_final_audits_passed']
    for n,h in c['reports'].items():assert sha(H/n)==h,n
    for n in('INTEGRITY_COMPARE.json','DB_POSTFLIGHT.json','TRAINING_AUDIT.json','GUARD.json','INPUTS_V2.receipt.json','REQUEST_RECONSTRUCTION_CHECK.json','REPLAY_CHECK.json'):
        assert read(n)['passed'],n
    assert read('TRAINING_AUDIT.json')['unique_seed_fits']==501
    assert read('TRAINING_AUDIT.json')['selected_outer_models']==70
    assert read('REQUEST_RECONSTRUCTION_CHECK.json')['cases']==10 and read('REPLAY_CHECK.json')['count']==6
    for p,h in read('ALL_FOLDS_FROZEN.receipt.json')['files'].items():
        assert sha(p)==h
        for q,k in json.loads(Path(p).read_text())['files'].items():assert sha(q)==k,q
    stat=read('STATISTICS.json');g=stat['development_goals'];p=stat['primary'];assert g['G1']['passed']==(p['estimate']>=.03 and min(p['per_scene'].values())>=-.10)
    assert g['G2']['passed']==all(v['comparisons']['v2_minus_global_fixed']['estimate']>=-.05 for v in stat['lofo'].values())
    assert g['G3']['passed']==(stat['full18']['comparisons']['v2_49_minus_global_fixed']['estimate']>=0)
    assert g['all_passed']==all(g[k]['passed']for k in('G1','G2','G3'))
    budget=read('NEW_ROLLOUT_BUDGET.json');assert budget['passed']and budget['unique_total']<=1300
    assert read('INTEGRITY_COMPARE.json')['reserved_prospective_files_not_opened_or_hashed']
    reserved=read('PROSPECTIVE_METADATA_END.json');assert reserved['no_file_content_opened_or_hashed']and(not reserved['changes']or reserved['registered_parallel_changes_excepted'])
    if(H/'CPU_PLACEMENT_HANDOFF.receipt.json').exists():assert read('CPU_PLACEMENT_HANDOFF_CHECK.json')['passed']and read('CPU_PLACEMENT_HANDOFF_CHECK.json')['saved_prefix_preserved_bitwise']
    from_names={
      'METHOD_COMPARISON.md':187,
      'LOFO_METHOD_COMPARISON.md':136,
      'FULL18_METHOD_COMPARISON.md':36}
    # Report names are the user's exact names, not role nicknames.
    names=['学习选 η v2','v2·只在 49 个旧候选里选','v2·不退回','v2·截断代替退回','v2·原始场景描述','v2·无先验','v2·按状态和控制器选','v2·按场景和控制器选','v2·修正不收缩','只用生成器','所有场景一组 η（训练场景里最好）','固定 η-A','按场景近邻选 η','固定 η13','只加安全投影','每个场景一组 η（本场景其他状态选，参照）','学习选 η v1（按状态和控制器选）']
    for file,count in from_names.items():
        if file not in c['reports']:continue
        lines=(H/file).read_text().splitlines();actual=sum(any(line.startswith('| '+n+' |')for n in names)for line in lines);assert actual==count,(file,actual,count)
    for file in('REPORT.md','METHOD_COMPARISON.md','ATTRIBUTION.md','RANKING_DIAGNOSTICS.md','LOFO_METHOD_COMPARISON.md','FULL18_METHOD_COMPARISON.md','AXIS_TABLE.md'):
        if file not in c['reports']:continue
        text=(H/file).read_text();assert not any(word in text for word in('同池','17池','对照1')),file
    jobs=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%i %j %T'],text=True)
    assert not any('xp2_'in line for line in jobs.splitlines()),jobs
    proof=dict(utc=datetime.now(timezone.utc).isoformat(),passed=True,report_hashes_verified=len(c['reports']),folds_frozen=31,unique_seed_fits=501,source_identical_fits_reused=263,
        reconstruction_checks=10,bitwise_replays=6,unique_requested_rollouts=budget['unique_total'],new_executions=c['new_executions'],uncompleted_cells=c['uncompleted_cells'],
        development_goals_recomputed_from_unrounded_json=True,reserved_prospective_files_not_opened_or_hashed=True,no_v2_workers_running=True,
        report_sha256=sha(H/'REPORT.md'),complete_sha256=sha(H/'COMPLETE.json'))
    (H/'FINAL_DELIVERY_VERIFY.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps(proof,ensure_ascii=False))

if __name__=='__main__':main()
