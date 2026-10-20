from core import *
from design import localdb
localdb()
from shared_rollout_db.src.rollout_db import uid,canonical,connect
from shared_rollout_db.src import historical_ingest
P=read(HERE/'protocol.json')
class ExactIngestor(historical_ingest.Ingestor):
 def __init__(self):
  # Existing schema is the read-only snapshot; do not initialize shared DB.
  self.root=ROOT;self.con=connect();self.stats={k:0 for k in ['files_scanned','experiment_dirs','seed_imported','aggregate_imported','duplicates','conflicts','ambiguous','summary_only','rows_seen']};self.experiments=set()
 def insert_seed(self,r,path,source,experiment,line):
  assert experiment==EXP and r['experiment_uid']==EXP and not r['jax_enable_x64'];c=r['case'];v=P['controllers'][c['controller_name']]
  assert r['protocol_sha256']==sha(HERE/'protocol.json') and r['execution_code_sha256']==sha(HERE/'runtime.py') and r['controller_code_sha256']==sha(OLD/'controllers.py')
  assert v['controller_uid']==c['controller_uid'] and sha(r['trace_path'])==r['trace_sha256'];assert self.con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone()[0]==canonical(v['payload'])
  assert c['rollout_uid']==uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
  flags=('success','deadlock','timeout','collision','numerical_failure');values=tuple(r[k] for k in flags);old=self.con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
  if old:
   assert values==tuple(old[k] for k in flags) and old['episode_length']==r['episode_length'] and not old['conflict_quarantined'];self.stats['duplicates']+=1;status='duplicate'
  else:
   self.con.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*values,r['episode_length'],r['outcome'],EXP,str(path),'EXACT_REUSE',uid('record',r)));self.stats['seed_imported']+=1;status='imported'
  self.con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line));return status
if __name__=='__main__':
 prereg();assert read(HERE/'CONSISTENCY_GATE.json')['passed'];historical_ingest.Ingestor=ExactIngestor
 from shared_rollout_db.src import merger
 merger.Ingestor=ExactIngestor;merger.ROOT=HERE;sys.argv=['merger','--experiment',EXP];merger.main();write(HERE/'MERGE_RECEIPT.json',dict(utc=utc(),official_entry='shared_rollout_db.src.merger',DB=str(HERE/'rollout.sqlite'),single_process=True,global_DB_unmodified=True))
