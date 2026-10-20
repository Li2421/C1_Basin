from core import *
from audit_utils import source_inventory
from registration import connect

def main():
    assert read(HERE/'RUN_STATUS.json')['status']=='COMPLETE'
    for name in ['COMPONENT_TESTS.json','ANALYSIS_CHECKS.json','PAIRING_AUDIT.json','RHR_TRACE_AUDIT.json','JOURNAL_ONCE_AUDIT.json','HANDOFF_COMPLETION.json']:
        x=read(HERE/name);assert x.get('passed',x.get('status')=='PASS'),name
    mismatch=[]
    for p in HERE.glob('*.receipt.json'):
        for f,h in read(p).get('files',{}).items():
            if sha(f)!=h:mismatch.append((p.name,f))
    assert not mismatch,mismatch
    a=read(HERE/'INTEGRITY_COMPARISON.json');assert a['protected_content_pass'] and not a['changes'] and not a['removed_status_entries']
    result=read(HERE/'results.json');assert result['integrity_pass'] and result['DB_guard_pass'] and not result['resource_cap_pass']
    C=result['phases']['C'];decision=any(C['students'][k]['mean']>=.02 and C['students'][k]['CI_unknown_envelope'][0]>0 for k in ['controller','state']);assert decision==C['next_decentralized_critic']
    counts={}
    with connect(True) as db:
        for phase,total,new in [('A',1280,1280),('tune',4800,4800),('fair',960,960),('C',18240,13440)]:
            rs=read(HERE/f'{phase}_DB_ROWS.json');assert len(rs)==total and len(read(HERE/f'{phase}_MISSING_CASES.json'))==new;receipt=read(HERE/f'{phase}_MERGE_RECEIPT.json');assert receipt['db_exact'];counts[phase]=dict(total=total,new=new,unknown=receipt['unknown'])
        unknown_cases=[c for c in read(HERE/'C_CASES.json') if c['rollout_uid']=='roll_de314ea91e8d51bfc5bfe5179c2194f258479986e4043b77393343f53ac37a17'];assert len(unknown_cases)==1;u=unknown_cases[0];assert not (HERE/'C/raw'/f"{u['rollout_uid']}.json").exists();r=db.execute('SELECT numerical_failure FROM rollout WHERE rollout_uid=?',(u['rollout_uid'],)).fetchone();assert r[0]==1
    files={}
    for p in HERE.glob('imports_*.json'):
        for f,h in read(p)['imported_sources'].items():
            if Path(f).is_relative_to(HERE):continue
            assert sha(f)==h,f
            if f in files:assert files[f]==h
            files[f]=h
    source_inventory('imports_final_verify.json');write(HERE/'FINAL_VERIFICATION.json',dict(utc=utc(),scientific_delivery_pass=True,source_and_DB_guards_pass=True,all_receipt_hashes_pass=True,paired_design_pass=True,main_RHR_choice_frozen_before_labels=True,journal_exactly_once_pass=True,old_unknown_never_rerun=True,resource_historical_compliance_pass=False,resource_exception='Historical21CPU observation exceeded then-authorized20; subsequent monitoring gap means full peak/duration unknown. Later user authorization allCPU when idle preserved separately.',phases=counts,decisions_recomputed_pass=True,existing_imported_sources=files))
    seal('FINAL_VERIFICATION.receipt.json',[HERE/f for f in ['FINAL_VERIFICATION.json','DELIVERY.receipt.json','REPORT.md','HANDOFF.md','results.json','HANDOFF_COMPLETION.json','JOURNAL_ONCE_AUDIT.json','final_verify.py']],scientific_delivery_pass=True,resource_historical_compliance_pass=False)
    print(json.dumps(dict(status='COMPLETE',source_pass=True,DB_pass=True,resource_history_exception=True,C=C['students'],next_critic=C['next_decentralized_critic']),indent=2))
if __name__=='__main__':main()
