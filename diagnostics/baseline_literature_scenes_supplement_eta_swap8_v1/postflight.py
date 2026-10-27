import bootstrap
from experiment import *
from shared_rollout_db.src.rollout_db import connect,canonical as dbcanonical
from registry import EXP
from evidence_store import load,locate
if __name__=='__main__':
    errors=[];seen=set();trace_seen=set();unknown=0;reused=0;new=0;sources=[]
    with connect(True) as db:
        for manifest in sorted(HERE.glob('*_CASES.json')):
            stage=manifest.name[:-len('_CASES.json')]
            if not (HERE/f'{stage}_REGISTRATION.receipt.json').exists():continue
            for c in read(manifest):
                rid=c['rollout_uid']
                if rid in seen:continue
                seen.add(rid);r=load(stage,c);p=locate(stage,c)
                stored=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
                if not stored or stored['conflict_quarantined']:errors.append(dict(rollout_uid=rid,reason='missing_or_conflicted_db'));continue
                if any(stored[k]!=r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key','success','collision','timeout','numerical_failure','outcome')):errors.append(dict(rollout_uid=rid,reason='identity_or_outcome_mismatch'))
                # Reused records remain byte-identical old provenance; local aliases carry their own journal hash.
                if stored['raw_record_hash']!=hashlib.sha256(dbcanonical(r).encode()).hexdigest():errors.append(dict(rollout_uid=rid,reason='raw_hash_mismatch'))
                if c.get('reuse_source_sha256') and sha(p)!=c['reuse_source_sha256']:errors.append(dict(rollout_uid=rid,reason='historical_raw_changed'))
                trace=r['trace_path']
                if trace not in trace_seen:
                    trace_seen.add(trace)
                    if sha(trace)!=r['trace_sha256']:errors.append(dict(rollout_uid=rid,reason='trace_hash_mismatch'))
                unknown+=r['numerical_failure'];reused+=not p.resolve().is_relative_to(HERE);new+=p.resolve().is_relative_to(HERE)
                sources.append(dict(rollout_uid=rid,path=str(p),sha256=sha(p),trace=trace,trace_sha256=r['trace_sha256']))
    pending=list((ROOT/'shared_rollout_db/journals'/EXP).glob('*.jsonl'));proof=dict(utc=utc(),passed=not errors and not pending,distinct_requests=len(seen),new_local_records=new,historical_records=reused,unknowns_preserved=unknown,trace_hashes_checked=len(trace_seen),pending_journals=[str(p) for p in pending],errors=errors)
    write(HERE/'DB_POSTFLIGHT.json',proof);write(HERE/'EVIDENCE_SOURCES_END.json',sources);assert proof['passed'],proof
    print('Postflight passed',len(seen),flush=True)
