"""Remove explicitly invalid/quarantined sources after migration; originals remain untouched."""
import json
from .rollout_db import connect,transaction

MARKERS=('invalid_feature','invalid_cache','feature_mismatch','quarantine_batchshape','quarantine_debug','quarantined_pretransition','cancelled','canceled','excluded_rollout','quarantined_bad')
def main():
 with connect() as con, transaction(con):
  bad=[dict(r) for r in con.execute('SELECT source_uid,path FROM source_file') if any(x in r['path'].lower() for x in MARKERS)]
  source_ids=[r['source_uid'] for r in bad];paths=[r['path'] for r in bad]
  affected=set()
  for p in paths:
   for r in con.execute('SELECT conflict_uid,existing_json FROM conflict WHERE source_file=?',(p,)):
    try:affected.add(json.loads(r['existing_json'])['rollout_uid'])
    except:pass
   con.execute('DELETE FROM conflict WHERE source_file=?',(p,))
  for su in source_ids:con.execute('DELETE FROM rollout_source WHERE source_uid=?',(su,))
  orphans=[r[0] for r in con.execute('SELECT rollout_uid FROM rollout r WHERE NOT EXISTS (SELECT 1 FROM rollout_source s WHERE s.rollout_uid=r.rollout_uid)')]
  for rid in orphans:con.execute('DELETE FROM rollout WHERE rollout_uid=?',(rid,))
  for rid in affected:
   # Remaining conflicts quote the stable rollout_uid in existing_json.
   if not con.execute('SELECT 1 FROM conflict WHERE existing_json LIKE ? LIMIT 1',(f'%{rid}%',)).fetchone():con.execute('UPDATE rollout SET conflict_quarantined=0 WHERE rollout_uid=?',(rid,))
  for su in source_ids:con.execute("UPDATE source_file SET classification='SUMMARY_ONLY',rows_imported=0 WHERE source_uid=?",(su,))
 print(json.dumps({'excluded_sources':len(source_ids),'orphan_rollouts_removed':len(orphans),'conflicted_rollouts_rechecked':len(affected)}))
if __name__=='__main__':main()
