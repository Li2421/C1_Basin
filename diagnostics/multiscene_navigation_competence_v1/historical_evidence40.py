"""Reuse audited exact menu labels and recompute descriptors from existing traces."""
from common import *
from shared_rollout_db.src.rollout_db import connect,uid,canonical
from probe_measurements40 import behavior_statistics
from collections import Counter

NOMINAL_FIELDS=['flow_checkpoint_sha256','nominal_implementation_sha256','normalization_sha256','Gaussian_config_sha256','controller_class','observation_frame','source_Flow_recipe','nominal_parameters']
def nominal_key(profile):return uid('nominal',{k:profile.get(k) for k in NOMINAL_FIELDS})
def allowed(path):
    p=Path(path).resolve()
    for part in p.parts:
        assert not part.startswith('decentralized_') and not part.startswith('teacher_student_'),'Forbidden historical source path'
    return p

def main():
    previous=D/'multiscene_generalization_v1';source=previous/'HISTORICAL_COMPATIBLE_LABELS.json';old=read(source);accepted=[];excluded=[]
    with connect(True) as db:
        for row in old:
            profile=row['profile'];text=canonical(profile)
            if 'decentralized_' in text or 'teacher_student_' in text:excluded.append(dict(rollout_uid=row['rollout_uid'],reason='Pruned parallel-directory provenance'));continue
            cached=dict(db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(row['rollout_uid'],)).fetchone())
            assert all(cached[k]==row[k] for k in ['success','collision','timeout','numerical_failure','state_uid','controller_uid','eta_uid','seed_key','raw_record_hash']) and not cached['conflict_quarantined']
            row=dict(row,nominal_key=nominal_key(profile));accepted.append(row)
    keys={r['nominal_key'] for r in accepted};features={};index=[];incomplete=[];configs={}
    for row in accepted:
        key=row['nominal_key'];metadata=json.loads(row['goals_geometry_json']);cfg=metadata.get('config',metadata)
        if key in configs:assert all(configs[key][k]==cfg[k] for k in ['agent_radius','max_speed'])
        configs[key]=cfg
    for folder_name in ['axis2_controller_v1','other_controller_classes_v1','blackbox_families_v1']:
        folder=D/folder_name;ip=folder/'R5_RAW_INDEX.json'
        if not ip.exists():continue
        protocols={}
        for path in folder.glob('probe*protocol.json'):
            protocol=read(path)
            for name,value in protocol.get('profiles',{}).items():
                payload=value.get('payload',value);key=nominal_key(payload)
                if key in keys:protocols[name]=(key,payload,protocol)
        for entry in read(ip):
            rp=allowed(entry['raw_path']);r=read(rp);name=r['case'].get('base_controller_name')
            if name not in protocols:continue
            key,payload,protocol=protocols[name]
            assert sha(rp)==entry['raw_sha256'] and not r['terminal_labels_read'] and not any(k in r for k in ['success','outcome','termination'])
            tp=allowed(r['trace_path']);assert sha(tp)==r['trace_sha256'];z=np.load(tp)
            assert len(z['actions'])<=200
            if len(z['actions'])!=200:incomplete.append(str(rp));continue
            # All compatible historical Ring profiles use the exact original radius/speed.
            config=configs[key]
            values=behavior_statistics(z['positions'],z['actions'],z['projection_active'],z['goals'],config['agent_radius'],config['max_speed'])
            features.setdefault(key,[]).append(values);index.append(dict(raw_path=str(rp),raw_sha256=sha(rp),trace_path=str(tp),trace_sha256=sha(tp),nominal_key=key,old_statistics_recomputed=True))
    means={}
    for key in keys:
        values=features.get(key,[])
        # Require the full original16-state/two-eta design rather than partial means.
        means[key]=np.mean(values,axis=0) if len(values)==32 else [np.nan]*5
    p=HERE/'HISTORICAL_COMPATIBLE_LABELS40.json';write(p,accepted)
    q=HERE/'HISTORICAL_PROBE_FEATURES40.json';write(q,dict(features=means,index=index,incomplete_preserved=incomplete,missing_full_descriptor_keys=[k for k in keys if len(features.get(k,[]))!=32],missing_imputed_in_source_fold_only=True))
    summary=dict(utc=utc(),reused_source_path=str(source),reused_source_sha256=sha(source),rows=len(accepted),by_arm=dict(Counter(r['arm'] for r in accepted)),by_scene=dict(Counter(r['scene'] for r in accepted)),unknown_preserved=sum(r['numerical_failure'] for r in accepted),new_historical_rollouts=0,new_historical_training=0,nominal_instances=len(keys),complete_recomputed_descriptors=sum(len(features.get(k,[]))==32 for k in keys),excluded=excluded,old_audit=read(previous/'HISTORICAL_ACTION_COMPATIBILITY.json'),missing_scenes=['double_bottleneck','four_way_intersection'],missing_scene_reason='Round34 labels use flow-internal correction; current menu corrects after safety projection, so exact action compatibility is absent')
    s=HERE/'HISTORICAL_EVIDENCE40.json';write(s,summary)
    seal('HISTORICAL_EVIDENCE40_FROZEN.receipt.json',[p,q,s,HERE/'historical_evidence40.py'],before_generated_correction_evaluation_labels=True,source_G_rows=0)
    print(summary,flush=True)

if __name__=='__main__':main()
