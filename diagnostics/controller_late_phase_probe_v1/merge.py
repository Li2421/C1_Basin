"""Official single merger entry with an exact schedule journal adapter.

The legacy ingest parser drops seed namespaces and physical timestep. This local
adapter preserves already registered modern identities; old code is untouched.
"""
from core import *
from shared_rollout_db.src.rollout_db import uid,canonical
import runpy,time
from shared_rollout_db.src import historical_ingest

class ExactScheduleIngestor(historical_ingest.Ingestor):
    def insert_seed(self,r,path,source,experiment,line):
        if not hasattr(self,'schedule_protocol'):
            self.schedule_protocol=read(HERE/'protocol.json');self.schedule_protocol_sha=sha(HERE/'protocol.json')
        assert experiment==EXP and r['experiment_uid']==EXP and r['protocol_sha256']==self.schedule_protocol_sha
        p=self.schedule_protocol;c=r['case'];assert all(r[k]==c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
        assert r['execution_code_sha256']==p['runner_sha256'] and r['jax_enable_x64'] is False
        ctl=p['controllers']['ring_exchange'][c['controller_name']];assert ctl['controller_uid']==c['controller_uid']
        stored=self.con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone();assert stored and stored[0]==canonical(ctl['payload'])
        state=self.con.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone();s=next(x for x in p['states'] if x['state_uid']==c['state_uid']);assert state and state[0]==canonical(s['physical'])
        assert uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))==c['rollout_uid']
        flags=('success','deadlock','timeout','collision','numerical_failure');values=tuple(r[k] for k in flags);assert not(r['success'] and (r['collision'] or r['numerical_failure']))
        old=self.con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
        if old:
            assert values==tuple(old[k] for k in flags) and r['episode_length']==old['episode_length'] and not old['conflict_quarantined'];self.stats['duplicates']+=1;status='duplicate'
        else:
            self.con.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*values,r['episode_length'],r['outcome'],EXP,str(path),'EXACT_REUSE',uid('record',r)));self.stats['seed_imported']+=1;status='imported'
        self.con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line));return status

def main():
    prereg();historical_ingest.Ingestor=ExactScheduleIngestor
    sys.argv=['shared_rollout_db.src.merger','--experiment',EXP]
    from shared_rollout_db.src import merger
    merger.ROOT=HERE
    merger.main()
    write(HERE/'merge_audits'/f'{time.time_ns()}.json',dict(utc=utc(),official_entry='shared_rollout_db.src.merger',local_adapter='ExactScheduleIngestor: registered full seed_key/physical timestep preserved; no legacy identity rewrite',experiment=EXP))
if __name__=='__main__':main()
