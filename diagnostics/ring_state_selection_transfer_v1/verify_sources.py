"""Audit original exported outcomes against exact shared DB rows; no original file writes."""
import sqlite3,json
from collections import Counter
import numpy as np
from study import OUT,OLD,CONF,read,write,sha

def main():
 db=sqlite3.connect('file:'+str(OUT.parents[1]/'shared_rollout_db/rollout.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;uids=set();stats=Counter();unknown=Counter();discrepancies=[]
 sources=[(OLD/'cases.json',OLD/'source_dataset.npz','source'),(CONF/'cases.json',CONF/'fresh_rollouts.npz','development')]
 for cp,dp,name in sources:
  a=np.load(dp);arr=a['outcomes'];present=a['present']
  for c in read(cp):
   i,j,k=c['state_index'],c['eta_index'],c['seed']
   if not present[i,j,k]:continue
   r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert r is not None
   assert all(r[z]==c[z] for z in ['state_uid','eta_uid','controller_uid','seed_key'])
   expected=(int(r['success']),int(r['numerical_failure']));actual=tuple(int(z) for z in arr[i,j,k,:2]);
   if expected!=actual or r['conflict_quarantined']:discrepancies.append(dict(rollout_uid=c['rollout_uid'],expected=expected,actual=actual,quarantine=r['conflict_quarantined']))
   uids.add(c['rollout_uid']);stats[name]+=1;unknown[name]+=int(r['numerical_failure'])
 allold=set(uids)
 for f in ['C0_ROLLOUT_INDEX.json','C3_LEGACY_ROLLOUT_INDEX.json','c1/cached_rollout_uids.json']:allold.update(read(OUT/f))
 write(OUT/'REUSED_ROLLOUT_INDEX.json',sorted(allold));write(OUT/'N1_ROLLOUT_INDEX.json',sorted(uids));write(OUT/'SOURCE_VALIDATION.json',dict(status='PASS' if not discrepancies else 'FAIL',N1_unique=len(uids),N1_by_source=dict(stats),N1_unknown=dict(unknown),discrepancies=discrepancies,all_reused_unique=len(allold),catalog='C0 inventory predates C1 insertion; C3 asset audit uses frozen catalog',source_archives=[dict(file=str(f),sha256=sha(f)) for pair in sources for f in pair[:2]],DB_path=str(OUT.parents[1]/'shared_rollout_db/rollout.sqlite')))
 assert not discrepancies;print('PASS',dict(stats),dict(unknown),'allreused',len(allold))
if __name__=='__main__':main()
