from core import *
import subprocess
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

def main():
    prereg();labels_allowed();gate();assert len(list((HERE/'main/raw').glob('*.json')))==3072
    try:
        for script in ['merge.py','evaluate.py','report.py']:
            q=subprocess.run([PY,'-B',str(HERE/script)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(HERE/(script+'.log')).write_text(q.stdout);assert q.returncode==0,(script,q.stdout[-3000:])
        result=read(HERE/'results.json');assert result['postflight']['status']=='PASS'
        for row in read(HERE/'HISTORICAL_ASSET_INDEX.json')['assets']:assert sha(ROOT/row['path'])==row['sha256']
        reg=read(HERE/'PREREGISTRATION.receipt.json');rf=read(HERE/'R6_FROZEN.receipt.json');sf=read(HERE/'SELECTOR_FROZEN.receipt.json');endpoint=read(HERE/'BLEND_ENDPOINT_OPENING.receipt.json');assert reg['utc']<rf['utc']<sf['utc']<endpoint['utc'];assert sha(HERE/'SELECTOR_MODELS.json')==sf['models_sha256'];assert sha(HERE/'R6_FEATURES.npz')==rf['features_sha256']
        assert read(HERE/'PROBE_REPRODUCTION_AUDIT.json')['status']=='PASS'
        assert read(HERE/'INDEPENDENT_MODEL_AUDIT.json')['status']=='PASS'
        components=read(HERE/'CORE_COMPONENT_AUDIT.json')
        for path,entry in components['checks'].items():assert sha(Path(path))==entry['expected'],path
        with connect(True) as db:
            counts={}
            for exp in [EXP,EXP+'_unlabeled']:
                rows=db.execute('SELECT count(*) AS n,sum(numerical_failure) AS unknown,sum(conflict_quarantined) AS quarantine FROM rollout WHERE experiment_uid=?',(exp,)).fetchone();counts[exp]=dict(rows)
            assert counts[EXP]['n']==3104 and counts[EXP]['quarantine']==0;assert counts[EXP+'_unlabeled']['n']==0
            probe_sources=db.execute("SELECT sum(rows_seen),sum(rows_imported) FROM source_file WHERE experiment_uid=? AND classification='UNLABELED_SHORT_PROBE'",(EXP+'_unlabeled',)).fetchone();assert probe_sources[0]==416 and probe_sources[1]==0
        with connect() as db,transaction(db):
            metadata=json.loads(db.execute('SELECT metadata_json FROM experiment WHERE experiment_uid=?',(EXP,)).fetchone()[0])
            metadata.update(status='COMPLETE',main_rollouts=3072,gate_replays=32,fixed_records_reused=3072,H_L=result['H_L'],H_transfer=result['H_transfer'],unknown_preserved=result['postflight']['unknown'])
            db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,new_rollout_count=?,reused_rollout_count=?,metadata_json=? WHERE experiment_uid=?',(3104,3072,canonical(metadata),EXP))
        write(HERE/'FINAL_AUDIT.json',dict(status='PASS',historical_asset_hashes_unchanged=True,R6_and_selectors_hashes_unchanged=True,preregister_freeze_endpoint_chronology=True,DB_counts=counts,probe_source_rows=416,probe_success_rows=0,native_core_components_unchanged=True,video_requirement='Exempt pure Q evaluation, preregistered',CPU_worker_cap=17,GPU_workers=0))
        result['status']='COMPLETE';write(HERE/'results.json',result);files=['PREREGISTRATION.md','R6_FEATURES.npz','R6_FROZEN.receipt.json','SELECTOR_MODELS.json','STAGE_B.json','main_POSTFLIGHT.json','results.json','REPORT.md','HANDOFF.md','FINAL_AUDIT.json','L_PREDICTIONS.png','BLEND_REGRET.png']
        write(HERE/'COMPLETE.receipt.json',dict(utc=utc(),status='COMPLETE',artifacts={f:sha(HERE/f) for f in files},new_full_rollouts=3072,gate_replays=32,unlabeled_paired_probes=416,H_L=result['H_L'],H_transfer=result['H_transfer'],H_transfer_E_only=result['H_transfer_E_only'],video_exemption='Pure Q evaluation preregistered under AGENTS.md'))
        print(json.dumps(dict(status='COMPLETE',H_L=result['H_L'],H_transfer=result['H_transfer'],H_transfer_E_only=result['H_transfer_E_only'])))
    except Exception as exc:
        write(HERE/'FINALIZE_ERROR.json',dict(utc=utc(),error=repr(exc)));raise
if __name__=='__main__':main()
