"""Official journal repair from existing raw only, merger, exact postflight."""
from experiment import *
from registry import EXP
from shared_rollout_db.src import rollout_db as dbm,planner,cache_writer
from db_merge import main as merge
import shared_rollout_db
from closure_scope import STOP_REASON

def main():
    assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT/'shared_rollout_db')
    assert (cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
    merge();recovered=[];seen=set();expected={}
    stagefiles=[p for p in HERE.glob('*_CASES.json') if not p.name.endswith('_MISSING_CASES.json')]
    for p in stagefiles:
        for c in read(p):expected[c['rollout_uid']]=c
    with dbm.connect(True) as db:
        for p in HERE.glob('*/raw/*.json'):
            r=read(p);rid=r['rollout_uid']
            if rid in seen:continue
            seen.add(rid);assert rid in expected,(p,rid)
            c=expected[rid];assert all(r[k]==c[k] for k in ('state_uid','controller_uid','eta_uid','seed_key'))
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(rid,)).fetchone():continue
            assert sha(r['trace_path'])==r['trace_sha256'],p
            cache_writer.append_journal([r],EXP,worker_id='final_existing_raw_recovery')
            recovered.append(dict(rollout_uid=rid,raw=str(p),sha256=sha(p),new_physical_steps=0))
    if recovered:merge()
    stages=[]
    for p in stagefiles:
        stage=p.name[:-len('_CASES.json')];manifest=HERE/f'{stage}_PLANNED_ROLLOUTS.json';post=planner.preflight(manifest);write(HERE/f'{stage}_POSTFLIGHT.json',post)
        obsolete=stage=='capability_lit_narrow_passage'
        user_stopped=stage=='safety_lit_circle_five'
        absent=[];unknown=[];conflicts=[]
        with dbm.connect(True) as db:
            for c in read(p):
                row=db.execute('SELECT state_uid,controller_uid,eta_uid,seed_key,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                if row is None:absent.append(c['rollout_uid']);continue
                assert all(row[k]==c[k] for k in ('state_uid','controller_uid','eta_uid','seed_key'))
                if row['conflict_quarantined']:conflicts.append(c['rollout_uid'])
                elif row['numerical_failure']:unknown.append(c['rollout_uid'])
        unindexed_completed=[rid for rid in absent if rid in seen]
        stages.append(dict(stage=stage,obsolete_implementation_branch=obsolete,user_stopped=user_stopped,**post['summary'],indexed_unknown_not_retried=len(unknown),unindexed_identities=len(absent),unindexed_completed_identities=len(unindexed_completed),intentionally_unwritten_discarded=len(absent) if user_stopped else 0,stop_reason=STOP_REASON if user_stopped else None,quarantined_identities=len(conflicts)))
        # Official lookup intentionally excludes numerical_failure=1. Its
        # missing count is not a license to repeat a preserved unknown.
        assert not unindexed_completed,(stage,unindexed_completed)
        if not obsolete:assert (user_stopped or not absent) and not conflicts and post['summary']['ambiguous']==0,(stage,stages[-1])
    write(HERE/'FINAL_JOURNAL_RECOVERY.json',recovered)
    write(HERE/'DB_POSTFLIGHT.json',dict(utc=utc(),experiment_uid=EXP,stages=stages,existing_raw_journals_recovered=len(recovered),new_physical_steps=0,all_active_stage_identities_indexed=True,all_completed_evidence_indexed=True,user_stopped_policy='Circle flow unwritten requests intentionally discarded by user; not unknown, not restored, not required as indexed rollout evidence. Completed raw and journals retained and indexed.',unknown_policy='Official exact lookup excludes numerical_failure=1. Those identities remain indexed unknown and are never retried; official missing counts are shown alongside actual unindexed counts.',obsolete_branch='Eight zero-step implementation-error outputs preserved; no further requests executed',posthoc_deadlock_field='Derived4a diagnoses live in SAFETY_BASELINE/GEOMETRY_CENSUS, not the legacy raw deadlock=0 field'))
    seal('DB_POSTFLIGHT.receipt.json',[HERE/'DB_POSTFLIGHT.json',HERE/'FINAL_JOURNAL_RECOVERY.json',HERE/'db_postflight.py',*[HERE/f"{s['stage']}_POSTFLIGHT.json" for s in stages]],official_merger_only=True,no_preexisting_registration_rows_updated=True)

if __name__=='__main__':main()
