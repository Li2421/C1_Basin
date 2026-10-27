"""Final saved-evidence and official-index checks; no new closed-loop work."""
from core import *
import collections,datetime

def main(rows,frozen,labels):
    attempts={}
    for p in(HERE/'attempts').glob('*.json'):
        r=read(p);attempts.setdefault(r['rollout_uid'],[]).append(float(r['time']))
    heldout={x['uid']for ll in labels.values()for x in ll}
    if(HERE/'paper_holdout_planned.json').exists():heldout|={c['rollout_uid']for c in read(HERE/'paper_holdout_planned.json')['exact_configs']}
    earliest=min((t for uid in heldout for t in attempts.get(uid,[])),default=None)
    family_path=HERE/'FAMILIES_FROZEN.json'
    assert earliest is not None and family_path.stat().st_mtime<=earliest,'Family freeze must precede heldout attempts'
    paper_heldout={u for u in heldout if u in rows and rows[u]['purpose']=='paper_holdout'}
    earliest_paper=min((t for uid in paper_heldout for t in attempts.get(uid,[])),default=None)
    if earliest_paper is not None:
        assert(HERE/'PAPER_PARAMETERS_FROZEN.json').stat().st_mtime<=earliest_paper
    unavailable={e['type_id']for e in frozen['entries']if'scene'not in e}
    invalid_primary=[uid for uid in heldout if uid in rows and rows[uid]['type_id']in unavailable]
    reserved=[r['rollout_uid']for r in rows.values()if r['state_index']in range(8,12)]
    missing_traces=[r['rollout_uid']for r in rows.values()if not Path(r['trace']).is_file()]
    unknown_retries=[];db_errors=[]
    with db.connect(True)as con:
        for uid,r in rows.items():
            saved=con.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
            if not saved or saved['success']!=r['success']or saved['numerical_failure']!=r['numerical_failure']or saved['conflict_quarantined']:db_errors.append(uid)
            if r['numerical_failure']:
                raw_time=(HERE/'raw'/f'{uid}.json').stat().st_mtime
                if any(t>raw_time+1. for t in attempts.get(uid,[])):unknown_retries.append(uid)
    assert not invalid_primary and not reserved and not missing_traces and not db_errors and not unknown_retries
    attempt_count=sum(map(len,attempts.values()));assert attempt_count<=45000
    archive=HERE/'scene_revisions/v1_before_invalidating_type2/SCENES_FROZEN.json'
    unchanged=[]
    if archive.exists():
        prior={e['type_id']:e['scene']for e in read(archive)['entries']if'scene'in e}
        for e in frozen['entries']:
            if'scene'not in e:continue
            assert digest(e['scene'])==digest(prior[e['type_id']])
            unchanged.append(e['type_id'])
    output=dict(passed=True,complete_records=len(rows),attempts=attempt_count,hard_cap=45000,official_db_raw_mismatches=db_errors,missing_trace_files=missing_traces,reserved_state_rollouts=reserved,invalid_geometry_primary_holdout=invalid_primary,completed_unknown_later_attempts=unknown_retries,family_freeze_sha256=sha(family_path),family_freeze_mtime=family_path.stat().st_mtime,earliest_heldout_attempt=earliest,earliest_paper_heldout_attempt=earliest_paper,unchanged_valid_scene_types=unchanged,source_scene_freeze_time=frozen['freeze_time'],physical_split_note='Only states0..3 tune/select. States4..7 heldout after frozen families. States8..11 unused. Exact initial-state RNG key certification uses input-only arithmetic and creates no labels.',scope='All local raw records have matching official indexed outcomes and trace files. Existing per-record trace hashes were computed at write time; this audit does not rehash every large trace.')
    dump(HERE/'DELIVERY_AUDIT.json',output);return output
