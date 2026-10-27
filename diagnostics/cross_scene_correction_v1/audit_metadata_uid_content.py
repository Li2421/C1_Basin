"""Read-only comparison of every planned metadata value, including repeated UIDs."""
from common import *
from collections import Counter


def main():
    cache={};counts=Counter();stage_counts={};source_hashes={}
    with dbm.connect(True)as db:
        def check(table,key_column,key,values,context):
            ident=(table,key)
            if ident not in cache:
                row=db.execute(f'SELECT * FROM {table} WHERE {key_column}=?',(key,)).fetchone()
                assert row is not None,('METADATA_UID_MISSING',table,key,context)
                cache[ident]=dict(row)
            actual=cache[ident]
            differences={k:dict(expected=v,actual=actual[k])for k,v in values.items()if actual[k]!=v}
            assert not differences,('METADATA_UID_CONTENT_CONFLICT',table,key,context,differences)
            counts[table]+=1
        check('experiment','experiment_uid',EXP,dict(name='Cross scene correction v1',path=str(HERE),
            protocol_hash=sha(HERE/'PROTOCOL_AMENDMENT.md'),code_hash=sha(HERE/'common.py'),
            metadata_json=canonical(dict(evidence='DEVELOPMENT'))),'experiment')
        for stage in('replay','capability','pilot','main','extrapolation'):
            path=HERE/f'{stage}_CASES.json'
            if not path.exists():continue
            cases=read(path);stage_counts[stage]=len(cases)
            for c in cases:
                s=c['state'];physical={k:s[k]for k in('positions','velocities','goals')}
                context=dict(stage=stage,rollout_uid=c['rollout_uid'])
                check('scenario','scenario_uid',c['scenario_uid'],dict(name=s['scene']+'_'+c['version'],
                    code_config_fingerprint=c['scenario_uid'],metadata_json=canonical(c['scenario_payload'])),context)
                check('state','state_uid',c['state_uid'],dict(scenario_uid=c['scenario_uid'],
                    source_group=f"N{s['N']}_state{s['index']}",content_hash=uid('physical',physical),
                    physical_state_json=canonical(physical),goals_geometry_json=canonical(c['scenario_payload']),
                    provenance_json=canonical(c['state_provenance']),identity_quality='CONTENT_EXACT'),context)
                check('controller_config','controller_uid',c['controller_uid'],dict(scenario_uid=c['scenario_uid'],
                    config_json=canonical(c['controller_payload']),compatibility_quality='EXACT_PROFILE'),context)
                eid,eta,hx=dbm.eta_identity(c['eta_a']);assert eid==c['eta_uid']
                check('eta','eta_uid',eid,dict(eta1=eta[0],eta2=eta[1],eta3=eta[2],canonical_hex=hx),context)
            del cases
            source_hashes[str(path)]=sha(path)
            print('Metadata every-case content matched:',stage,stage_counts[stage],flush=True)
    write(HERE/'METADATA_UID_CONTENT_AUDIT.json',dict(utc=utc(),passed=True,read_only=True,
        stage_requests=stage_counts,all_occurrences_checked=dict(counts),unique_table_uid_rows=len(cache),
        includes_repeated_uids=True,files=source_hashes,no_database_writes=True,
        supplements='Registration transaction old-row before/after hashes; verifies every input occurrence, not only first UID occurrence.'))


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(HERE/'METADATA_UID_AUDIT_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc()))
        raise
