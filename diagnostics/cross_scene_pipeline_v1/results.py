"""Read completed local evidence or exact merged official cache records."""
from common import *

@lru_cache(None)
def result(stage,rid):
    for name in (stage,'pilot','main'):
        p=HERE/name/'raw'/f'{rid}.json'
        if p.exists():return read(p)
    with dbm.connect(True)as db:
        rows=db.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf USING(source_uid)WHERE rs.rollout_uid=?',(rid,)).fetchall()
    for row in rows:
        p=Path(row['path'])
        if not p.exists():continue
        with p.open()as f:r=next((json.loads(line)for i,line in enumerate(f,1)if i==row['source_line']),None)
        if r and r.get('rollout_uid')==rid:
            evidence(p);return r
    raise FileNotFoundError(rid)
