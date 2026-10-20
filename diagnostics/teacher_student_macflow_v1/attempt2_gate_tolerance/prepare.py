from core import *
import sqlite3
p=read(D/'decentralized_pipeline_v1/main_protocol.json')
manifest_paths=list((D/'ring_exchange_stage1').glob('*/manifest.json'))
manifests={sha(m):m for m in manifest_paths}
rows=[]
for i,n in enumerate(p['controller_order']):
    v=p['base_profiles'][n]['payload'];checkpoint=Path(v['flow_checkpoint_path']);assert sha(checkpoint)==v['flow_checkpoint_sha256']
    cfg_path=D/'ring_exchange_stage1'/n/'config.json';cfg=read(cfg_path)
    m=manifests.get(cfg['dataset_manifest_sha256'])
    # Training configs may preserve the original manifest digest before appended audit metadata.
    if m is None:
        source=D/'ring_exchange_stage1'/(n.removesuffix('_macflow')+'_dataset')/'manifest.json'
        if source.exists():m=source
    assert m is not None,(n,cfg['dataset_manifest_sha256'])
    rows.append(dict(name=n,index=i,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                     train_config_path=str(cfg_path),train_config_sha256=sha(cfg_path),dataset_manifest_path=str(m),
                     dataset_manifest_sha256=sha(m),training_manifest_digest=cfg['dataset_manifest_sha256'],
                     manifest_digest_matches_training=sha(m)==cfg['dataset_manifest_sha256'],student_seed=2026103000+i))
write(HERE/'TEACHERS.json',rows)
with sqlite3.connect('file:'+str(ROOT/'shared_rollout_db/rollout.sqlite')+'?mode=ro',uri=True) as db:
    historical=db.execute('SELECT state_uid,content_hash,physical_state_json FROM state').fetchall()
    metadata=db.execute("SELECT controller_uid,config_json FROM controller_config WHERE config_json LIKE '%student%' OR config_json LIKE '%onestep%' OR config_json LIKE '%distill%'").fetchall()
write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),historical_state_count=len(historical),student_metadata_candidates=metadata,
 sources=['attempt1_blocked/HISTORY_AUDIT.md','round28 main_protocol, EVAL_STATES, frozen pipeline and teacher checkpoint hashes',
          'round27 probes/freeze_choices/controllers','official agents/macflow.py','Toy/Double-Bottleneck Stage-I implementations'],
 reuse='Reuse round28 teacher four-arm three-pipeline labels and expert reference. Teacher nominal/safety require exact identity preflight. No compatible student checkpoint established.',
 gap='New own-observation one-step students and their execution semantics need new training and closed-loop evidence.',
 teacher_training_manifest_discrepancies=[r for r in rows if not r['manifest_digest_matches_training']],
 no_rollout_executed=True))
print('Teacher hashes PASS',len(rows),'manifest discrepancies',sum(not r['manifest_digest_matches_training'] for r in rows))
