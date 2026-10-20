from core import *
from shared_rollout_db.src.rollout_db import uid,canonical
from shared_rollout_db.src import historical_ingest
class ExactIngestor(historical_ingest.Ingestor):
 def ingest_jsonl(self,path,experiment_override=None,journal=False):
  path=Path(path);source,experiment,_=self.source(path,'SEED_EXACT_CANDIDATE',experiment_override=experiment_override);seen=imp=dup=reserved=0
  for line_no,line in enumerate(path.read_text().splitlines(),1):
   if not line.strip():continue
   seen+=1;self.stats['rows_seen']+=1;r=json.loads(line)
   if r.get('schema')=='RESERVATION_NOT_EVIDENCE':
    assert not any(k in r for k in ('success','outcome','numerical_failure'));assert r['plan_sha256'] in {sha(HERE/'RESOURCE_BOOST_PLAN.json'),sha(HERE/'RESOURCE_BOOST2_PLAN.json')};actual=read(HERE/'main/raw'/f"{r['case']['rollout_uid']}.json");assert actual['case']==r['case'] and actual.get('schema')!='RESERVATION_NOT_EVIDENCE';reserved+=1;continue
   status=self.insert_seed(r,path,source,experiment,line_no);imp+=status=='imported';dup+=status=='duplicate'
  self.con.execute('UPDATE source_file SET classification=?,rows_seen=?,rows_imported=?,rows_duplicate=?,rows_ambiguous=? WHERE source_uid=?',('SEED_EXACT' if imp or dup else 'SUMMARY_ONLY',seen,imp,dup,0,source));self.con.commit();self.stats['operational_reservations_skipped']=self.stats.get('operational_reservations_skipped',0)+reserved;return imp
 def insert_seed(self,r,path,source,experiment,line):
  assert experiment==EXP and r['experiment_uid']==EXP;c=r['case'];ppath=HERE/('gate_protocol.json' if 'expected_uid' in c else 'protocol.json');p=read(ppath);assert r['protocol_sha256']==sha(ppath) and r['execution_code_sha256']==p['runner_sha256'] and r['jax_enable_x64'] is False;ctl=p['controllers']['ring_exchange'][c['controller_name']];assert ctl['controller_uid']==c['controller_uid'];stored=self.con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone();assert stored[0]==canonical(ctl['payload']);assert uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))==c['rollout_uid'];flags=('success','deadlock','timeout','collision','numerical_failure');values=tuple(r[k] for k in flags);old=self.con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
  if old:assert values==tuple(old[k] for k in flags) and r['episode_length']==old['episode_length'] and not old['conflict_quarantined'];self.stats['duplicates']+=1;status='duplicate'
  else:self.con.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*values,r['episode_length'],r['outcome'],EXP,str(path),'EXACT_REUSE',uid('record',r)));self.stats['seed_imported']+=1;status='imported'
  self.con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line));return status
if __name__=='__main__':
 prereg();historical_ingest.Ingestor=ExactIngestor;from shared_rollout_db.src import merger;merger.ROOT=HERE;sys.argv=['merger','--experiment',EXP];merger.main();write(HERE/'MERGE_RECEIPT.json',dict(utc=utc(),official_entry='shared_rollout_db.src.merger',exact_seed_adapter_sha256=sha(__file__),workers_do_not_write_SQLite=True))
