"""Read-only provenance investigation of concurrent shared-database changes."""
from util import *
import sqlite3

def main():
    baseline=read(HERE/'DB_READ_ONLY_BASELINE.json')
    start=read(HERE/'PREREGISTRATION.receipt.json')['utc'][:19].replace('T',' ')
    path=Path(baseline['path'])
    before=path.stat()
    conn=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    conn.execute('PRAGMA query_only=ON')
    conn.row_factory=sqlite3.Row
    sources=[dict(x) for x in conn.execute('SELECT path,ingested_at,rows_imported,experiment_uid FROM source_file WHERE ingested_at >= ? ORDER BY ingested_at',(start,))]
    experiments=[dict(x) for x in conn.execute('SELECT experiment_uid,name,path,start_time FROM experiment WHERE start_time >= ? ORDER BY start_time',(start,))]
    own=conn.execute('SELECT COUNT(*) FROM experiment WHERE path LIKE ?', (str(HERE)+'%',)).fetchone()[0]
    own_sources=conn.execute('SELECT COUNT(*) FROM source_file WHERE path LIKE ?', (str(HERE)+'%',)).fetchone()[0]
    conn.close()
    current=sha(path);after=path.stat()
    known_external=bool(sources and experiments) and all(not x['path'].startswith(str(HERE)) for x in sources+experiments)
    result=dict(utc=utc(),status='CONCURRENT_EXTERNAL_ACTIVITY_IDENTIFIED' if known_external and own==own_sources==0 else 'UNRESOLVED',connection='mode=ro&immutable=1; query_only=ON; SELECT only',baseline=baseline,current_sha256=current,current_bytes=after.st_size,current_mtime_ns=after.st_mtime_ns,stable_during_inspection=before.st_mtime_ns==after.st_mtime_ns,study_experiment_rows=own,study_source_file_rows=own_sources,external_sources_since_preregistration=sources,external_experiments_since_preregistration=experiments,source_rows_imported_since_preregistration=sum(x['rows_imported'] for x in sources),scope='Concurrent Drone3D provenance explains an expected mutable shared DB. No old DB snapshot exists to prove every byte delta; global database invariance is NOT VERIFIED. This study uses frozen NPZ/checkpoints, whose hashes are checked independently. The training/endpoint scripts never connect to DB; this final investigator opens read-only.')
    write(HERE/'DB_CONCURRENT_ACTIVITY.json',result)
    print(json.dumps({k:result[k] for k in ('status','study_experiment_rows','study_source_file_rows','source_rows_imported_since_preregistration')}))

if __name__=='__main__':main()
