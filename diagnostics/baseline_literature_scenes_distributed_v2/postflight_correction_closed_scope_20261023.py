"""Read-only exact-identity postflight after the single official final merge."""
from experiment import *
from registry import EXP,dbm,verify_receipt
from result_inventory import find_raw
from closed_old_circle_scope_20261023 import preserved_scope,OLD_STAGE

def main():
    prereg();verify_receipt('MIGRATION_NUMBERS.receipt.json')
    checked=set();differences=[];stages=[];physical_steps=0;traces={};own_raw=0
    with dbm.connect(True) as db:
        for proof in HERE.glob('*_PREFLIGHT.receipt.json'):
            stage=proof.name.removesuffix('_PREFLIGHT.receipt.json');verify_receipt(proof.name)
            cases=read(HERE/f'{stage}_CASES.json');complete=0;absent=[]
            closed=preserved_scope() if stage==OLD_STAGE else None
            for c in cases:
                rid=c['rollout_uid']
                if closed is not None and rid not in closed:
                    absent.append(rid);continue
                try:path=Path(closed[rid]['raw']) if closed is not None else find_raw(rid,stage,c['state']['scene'])
                except FileNotFoundError:
                    absent.append(rid);continue
                complete+=1
                if rid in checked:continue
                checked.add(rid);raw=read(path)
                row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                if row is None:differences.append(dict(kind='unindexed_completed_raw',rollout_uid=rid));continue
                expected={k:c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key')}
                expected.update({k:raw[k] for k in ('success','timeout','collision','numerical_failure')})
                if any(row[k]!=v for k,v in expected.items()) or row['conflict_quarantined']:
                    differences.append(dict(kind='cache_mismatch',rollout_uid=rid))
                trace=Path(raw['trace_path'])
                if trace not in traces:traces[trace]=sha(trace)
                if traces[trace]!=raw['trace_sha256']:differences.append(dict(kind='trace_hash',path=str(trace)))
                if path.is_relative_to(HERE):
                    own_raw+=1;physical_steps+=raw.get('new_physical_steps',raw.get('episode_length',0))
            user_stopped=False
            if stage=='learning_labels_lit_circle_five':
                verify_receipt('OLD_CIRCLE_LEARNING_STOPPED_20261021.receipt.json')
                preserved=read(HERE/'OLD_CIRCLE_LEARNING_LANDED_20261021.json')['rows']
                old_uids={r['rollout_uid'] for r in preserved}
                expected_absent={c['rollout_uid'] for c in cases}-old_uids
                assert set(absent)==expected_absent and len(absent)==4100,(stage,len(absent))
                assert complete==2044
                for r in preserved:assert sha(r['raw'])==r['raw_sha256'],r['raw']
                user_stopped=True
            main_stage=not stage.startswith('optional_') and not user_stopped
            if main_stage and absent:differences.append(dict(kind='missing_main_requests',stage=stage,count=len(absent)))
            later_exact_eta0_cache_hits=0
            if closed is not None:
                for c in cases:
                    if c['rollout_uid'] in closed:continue
                    row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                    if row is not None:
                        assert c.get('menu_index')==0 and not any(c['eta_a']) and not any(c['eta_b']),'Unexpected old nonzero execution after stop'
                        later_exact_eta0_cache_hits+=1
            stages.append(dict(stage=stage,planned=len(cases),completed=complete,missing=len(absent),main=main_stage,user_stopped=user_stopped,unwritten_not_unknown=user_stopped,later_exact_eta0_cache_hits_not_old_stage_completion=later_exact_eta0_cache_hits,closed_original_raw_scope=closed is not None))
        quarantined=db.execute('SELECT COUNT(*) FROM rollout WHERE experiment_uid=? AND conflict_quarantined!=0',(EXP,)).fetchone()[0]
        if quarantined:differences.append(dict(kind='own_quarantined_rows',count=quarantined))
    pending=list((ROOT/'shared_rollout_db/journals'/EXP).glob('*.jsonl'))
    if pending:differences.append(dict(kind='unmerged_completed_journals',count=len(pending)))
    write(HERE/'DB_POSTFLIGHT.json',dict(utc=utc(),passed=not differences,differences=differences,stages=stages,unique_completed_identities=len(checked),unique_trace_files=len(traces),own_completed_identities=own_raw,new_physical_steps=physical_steps,own_quarantined_rows=quarantined,unmerged_complete_journals=len(pending)))
    seal('DB_POSTFLIGHT.receipt.json',[HERE/'DB_POSTFLIGHT.json',HERE/'postflight_correction_closed_scope_20261023.py'],passed=not differences)
    assert not differences,differences[:10]

if __name__=='__main__':main()
