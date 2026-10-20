from core import *
from shared_rollout_db.src.rollout_db import uid,canonical,connect
from shared_rollout_db.src import historical_ingest
from shared_rollout_db.src.planner import preflight
MAIN='exp_decentralized_pipeline_v1'; EXPERT=MAIN+'_expert'
PROTOCOLS={sha(HERE/f'{phase}_protocol.json'):read(HERE/f'{phase}_protocol.json') for phase in ['main','expert']}
class ExactIngestor(historical_ingest.Ingestor):
 def insert_seed(self,r,path,source,experiment,line):
  assert experiment in (MAIN,EXPERT) and r['experiment_uid']==experiment and not r['jax_enable_x64']
  p=PROTOCOLS[r['protocol_sha256']];c=r['case']
  if experiment==MAIN:
   v=p['controllers'][c['controller_name']];assert r['pipeline_code_sha256']==p['pipeline_sha256'];assert r['execution_code_sha256']==p['runner_sha256']
  else:v={'controller_uid':p['controller_uid'],'payload':p['payload']};assert r['execution_code_sha256']==p['expert_code_sha256']
  assert v['controller_uid']==c['controller_uid'];assert sha(r['trace_path'])==r['trace_sha256'];assert self.con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone()[0]==canonical(v['payload']);assert c['rollout_uid']==uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
  flags=('success','deadlock','timeout','collision','numerical_failure');values=tuple(r[k] for k in flags);old=self.con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
  if old:
   assert values==tuple(old[k] for k in flags) and old['episode_length']==r['episode_length'] and not old['conflict_quarantined'];self.stats['duplicates']+=1;status='duplicate'
  else:
   self.con.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*values,r['episode_length'],r['outcome'],experiment,str(path),'EXACT_REUSE',uid('record',r)));self.stats['seed_imported']+=1;status='imported'
  self.con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line));return status

def main():
 prereg();historical_ingest.Ingestor=ExactIngestor
 from shared_rollout_db.src import merger
 merger.ROOT=HERE;post={}
 for phase,exp,n in [('main',MAIN,12544),('expert',EXPERT,32)]:
  rows=[read(f) for f in sorted((HERE/phase/'raw').glob('*.json'))];assert len(rows)==n;assert len({r['case']['rollout_uid'] for r in rows})==n
  if phase=='main':
   workers=[read(HERE/'main'/f'worker_{i}.json') for i in range(18)];assert all(w['complete'] for w in workers) and sum(w['assigned'] for w in workers)==n
  sys.argv=['merger','--experiment',exp];merger.main()
  with connect(True) as db:
   records=db.execute('SELECT * FROM rollout WHERE experiment_uid=?',(exp,)).fetchall();assert len(records)==n and not any(r['conflict_quarantined'] for r in records)
   for r in rows:
    found=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['case']['rollout_uid'],)).fetchone();assert found and all(found[k]==r[k] for k in ('success','deadlock','timeout','collision','numerical_failure','episode_length'))
  pf=preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json');write(HERE/f'{phase}_CACHE_POSTFLIGHT.json',pf)
  missing={uid('roll',dict(state=d['state_uid'],eta=d['eta_uid'],controller=d['controller_uid'],seed=s)) for d in pf['details'] for s in d.get('missing_seeds',[])};unknown={r['case']['rollout_uid'] for r in rows if r['numerical_failure']};assert missing==unknown,(len(missing),len(unknown));assert not pf['summary']['ambiguous']
  post[phase]=dict(records=n,unknown=len(unknown),missing_equals_preserved_unknown=True,db_exact=True)
 write(HERE/'MERGE_RECEIPT.json',dict(utc=utc(),official_entry='shared_rollout_db.src.merger',adapter_sha256=sha(__file__),workers_did_not_write_SQLite=True,postflight=post));print(post)
if __name__=='__main__':main()
