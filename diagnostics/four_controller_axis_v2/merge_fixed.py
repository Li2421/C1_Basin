"""Official shared DB merger with local modern exact-seed identity adapter."""
from __future__ import annotations
import datetime,hashlib,json,runpy,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import uid,canonical
from shared_rollout_db.src import historical_ingest
EXP='exp_round19b_four_controller_axis_fixed_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class ExactFixedIngestor(historical_ingest.Ingestor):
    def insert_seed(self,r,path,source,experiment,line):
        if not hasattr(self,'p'):
            self.p=read(HERE/'FIXED_PROTOCOL.json');self.ph=sha(HERE/'FIXED_PROTOCOL.json')
            self.states={s['state_uid']:s for s in self.p['states']}
        assert experiment==EXP and r['experiment_uid']==EXP and r['protocol_sha256']==self.ph
        c=r['case'];assert all(r[k]==c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
        assert r['execution_code_sha256']==self.p['runner_sha256'] and r['jax_enable_x64'] is False
        ctl=self.p['profiles'][c['controller_name']];assert ctl['controller_uid']==c['controller_uid']
        stored=self.con.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(c['controller_uid'],)).fetchone()
        assert stored and stored[0]==canonical(ctl['payload'])
        s=self.con.execute('SELECT physical_state_json FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone()
        assert s and s[0]==canonical(self.states[c['state_uid']]['physical'])
        assert uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))==c['rollout_uid']
        flags=('success','deadlock','timeout','collision','numerical_failure');values=tuple(r[k] for k in flags)
        assert not(r['success'] and (r['collision'] or r['numerical_failure']))
        old=self.con.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
        if old:
            assert values==tuple(old[k] for k in flags) and r['episode_length']==old['episode_length'] and not old['conflict_quarantined']
            self.stats['duplicates']+=1;status='duplicate'
        else:
            self.con.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*values,
                 r['episode_length'],r['outcome'],EXP,str(path),'EXACT_REUSE',uid('record',r)))
            self.stats['seed_imported']+=1;status='imported'
        self.con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line))
        return status
def main():
    assert read(HERE/'R5_FREEZE.receipt.json')['features_sha256']==sha(HERE/'R5_FROZEN.npz')
    assert all(read(HERE/'stage2'/f'worker_{i:02d}.json')['complete'] for i in range(18))
    historical_ingest.Ingestor=ExactFixedIngestor
    from shared_rollout_db.src import merger
    merger.ROOT=HERE;sys.argv=['shared_rollout_db.src.merger','--experiment',EXP]
    merger.main()
    p=HERE/'FIXED_MERGE.receipt.json'
    p.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        official_entry='shared_rollout_db.src.merger',local_exact_identity_adapter_sha256=sha(__file__),
        experiment_uid=EXP),indent=2)+'\n')
if __name__=='__main__':main()
