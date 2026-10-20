"""Official single merger entry with an exact schedule journal adapter.

The legacy ingest parser drops seed namespaces and physical timestep. This local
adapter preserves already registered modern identities; old code is untouched.
"""
from core import *
from shared_rollout_db.src.rollout_db import uid,canonical
import runpy,time
from shared_rollout_db.src import historical_ingest

class ExactScheduleIngestor(historical_ingest.Ingestor):
    def ingest_jsonl(self,path,experiment_override=None,journal=False):
        path=Path(path);source,experiment,_=self.source(path,'SEED_EXACT_CANDIDATE',experiment_override=experiment_override);seen=imp=dup=amb=reserved=0
        for line_no,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip():continue
            seen+=1;self.stats['rows_seen']+=1;r=json.loads(line)
            if str(r.get('schema','')).startswith('RESERVATION_'):
                assert not any(k in r for k in ['success','outcome','termination','numerical_failure'])
                reservation_plan=Path(r.get('reservation_plan_path',str(HERE/'RESOURCE_BOOST_PLAN.json')));assert reservation_plan.is_relative_to(HERE);assert r['reservation_plan_sha256']==sha(reservation_plan)
                actual=read(HERE/'main/raw'/f"{r['case']['rollout_uid']}.json")
                assert actual['case']==r['case'] and actual['protocol_sha256']==r['protocol_sha256'] and not actual['schema'].startswith('RESERVATION_')
                reserved+=1;self.stats['operational_reservations_skipped']=self.stats.get('operational_reservations_skipped',0)+1;continue
            if not any(k in r for k in ['success','outcome','termination']):amb+=1;continue
            status=self.insert_seed(r,path,source,experiment,line_no)
            if status=='imported':imp+=1
            elif status=='duplicate':dup+=1
            else:amb+=1
        prior=self.con.execute('SELECT classification FROM source_file WHERE source_uid=?',(source,)).fetchone();classification='SEED_EXACT' if imp or dup or prior and prior['classification']=='SEED_EXACT' else 'OPERATIONAL_RESERVATIONS_ONLY' if reserved and not amb else 'SUMMARY_ONLY'
        self.con.execute('UPDATE source_file SET classification=?,rows_seen=?,rows_imported=?,rows_duplicate=?,rows_ambiguous=? WHERE source_uid=?',(classification,seen,imp,dup,amb,source));self.con.commit();return imp

    def insert_seed(self,r,path,source,experiment,line):
        if not hasattr(self,'schedule_protocol'):
            ppath=HERE/('gate_protocol.json' if experiment.endswith('__gate') else 'protocol.json');self.schedule_protocol=read(ppath);self.schedule_protocol_sha=sha(ppath)
        assert experiment in [EXP,EXP+'__gate'] and r['experiment_uid']==experiment and r['protocol_sha256']==self.schedule_protocol_sha
        p=self.schedule_protocol;c=r['case'];assert all(r[k]==c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
        profile_key=c.get('variant',c['scene']);allowed={p['run_sha256']}
        if (HERE/'RUN_IO_REPAIR.receipt.json').exists():allowed.add(read(HERE/'RUN_IO_REPAIR.receipt.json')['after_sha256'])
        assert r['execution_code_sha256'] in allowed and r['jax_enable_x64']==p['controllers'][profile_key][c['controller_name']]['payload']['jax_enable_x64']
        ctl=p['controllers'][profile_key][c['controller_name']];assert ctl['controller_uid']==c['controller_uid']
        stored=self.con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone();assert stored and stored[0]==canonical(ctl['payload'])
        state=self.con.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone();s=next(x for x in p['states'] if x['state_uid']==c['state_uid']);assert state and state[0]==canonical(s['physical'])
        assert uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))==c['rollout_uid']
        flags=('success','deadlock','timeout','collision','numerical_failure');values=tuple(r[k] for k in flags);assert not(r['success'] and (r['collision'] or r['numerical_failure']))
        old=self.con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
        if old:
            assert values==tuple(old[k] for k in flags) and r['episode_length']==old['episode_length'] and not old['conflict_quarantined'];self.stats['duplicates']+=1;status='duplicate'
        else:
            self.con.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*values,r['episode_length'],r['outcome'],experiment,str(path),'EXACT_REUSE',uid('record',r)));self.stats['seed_imported']+=1;status='imported'
        self.con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line));return status

def main():
    prereg();historical_ingest.Ingestor=ExactScheduleIngestor
    active=EXP+'__gate' if len(sys.argv)>1 and sys.argv[1]=='gate' else EXP
    sys.argv=['shared_rollout_db.src.merger','--experiment',active]
    from shared_rollout_db.src import merger
    merger.ROOT=HERE
    merger.main()
    write(HERE/'merge_audits'/f'{time.time_ns()}.json',dict(utc=utc(),official_entry='shared_rollout_db.src.merger',local_adapter='ExactScheduleIngestor: registered full seed_key/physical timestep preserved; no legacy identity rewrite',experiment=active))
if __name__=='__main__':main()
