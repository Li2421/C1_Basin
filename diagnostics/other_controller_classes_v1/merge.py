from core import *
from shared_rollout_db.src.rollout_db import uid,canonical,connect
from shared_rollout_db.src import historical_ingest
EXP='exp_other_controller_classes_v1'
PROTOCOLS={sha(HERE/f'{phase}_protocol.json'):read(HERE/f'{phase}_protocol.json') for phase in ['gate','pilot','main']}
class ExactIngestor(historical_ingest.Ingestor):
 def insert_seed(self,r,path,source,experiment,line):
  assert experiment==EXP and r['experiment_uid']==EXP and not r['jax_enable_x64'];p=PROTOCOLS[r['protocol_sha256']];c=r['case'];v=p['controllers'][c['controller_name']];assert v['controller_uid']==c['controller_uid'];assert r['controller_code_sha256']==p['controller_code_sha256'];ok=r['execution_code_sha256']==p['runner_sha256']
  if not ok:
   fix=read(HERE/'RUNNER_REPAIR.receipt.json');assert p['phase']=='gate' and r['execution_code_sha256']==fix['new_runner_sha256'] and p['runner_sha256']==fix['old_runner_sha256']
  assert sha(r['trace_path'])==r['trace_sha256'];assert self.con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone()[0]==canonical(v['payload']);assert c['rollout_uid']==uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']));flags=('success','deadlock','timeout','collision','numerical_failure');values=tuple(r[k] for k in flags);old=self.con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
  if old:
   assert values==tuple(old[k] for k in flags) and old['episode_length']==r['episode_length'] and not old['conflict_quarantined'];self.stats['duplicates']+=1;status='duplicate'
  else:
   self.con.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*values,r['episode_length'],r['outcome'],EXP,str(path),'EXACT_REUSE',uid('record',r)));self.stats['seed_imported']+=1;status='imported'
  self.con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line));return status
if __name__=='__main__':
 prereg();historical_ingest.Ingestor=ExactIngestor;from shared_rollout_db.src import merger;merger.ROOT=HERE;sys.argv=['merger','--experiment',EXP];merger.main();write(HERE/'MERGE_RECEIPT.json',dict(utc=utc(),official_entry='shared_rollout_db.src.merger',exact_adapter_sha256=sha(__file__),workers_did_not_write_SQLite=True))
