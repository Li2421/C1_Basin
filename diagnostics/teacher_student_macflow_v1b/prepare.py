from core import *
from audit_utils import source_inventory
from data import teacher
import jax
jax.config.update('jax_enable_x64',False)
OLD=D/'teacher_student_macflow_v1'
def main():
    assert (HERE/'INTEGRITY_START.json').exists()
    assert not (HERE/'PREREGISTRATION.receipt.json').exists()
    seal('PREREGISTRATION.receipt.json',[HERE/'PREREGISTRATION.md',ROOT/'docs/prompt_round30b_frame_consistent_students_20261015.md',HERE/'TEACHERS.json',HERE/'TRAINING_CONFIG.json'],closed_loop_started=False,training_started=False,cpu_limit=20,gpu_shards=2)
    old_sources=read(OLD/'GATE_IMPORTED_SOURCES.json')['imported_sources'];checked={}
    for p,h in old_sources.items():
        if '/teacher_student_macflow_v1/' in p:continue
        assert sha(p)==h,('noise gate source changed',p);checked[p]=h
    gate=read(OLD/'NOISE_GATE.json');assert gate['status']=='PASS' and gate['passed_samples']==1008
    write(HERE/'NOISE_GATE.json',dict(status='PASS',reuse=True,passed_samples=1008,source=str(OLD/'NOISE_GATE.json'),sha256=sha(OLD/'NOISE_GATE.json'),same_original_noise_implementation=True,source_hashes=checked))
    evidence=[];scales=[]
    for i in read(HERE/'GLOBAL_INDICES.json'):
        r=read(HERE/'TEACHERS.json')[i]
        for k in ['checkpoint','train_config_path','dataset_manifest_path']:
            hk={'checkpoint':'checkpoint_sha256','train_config_path':'train_config_sha256','dataset_manifest_path':'dataset_manifest_sha256'}[k];assert sha(r[k])==r[hk]
        folder=OLD/'data'/r['name'];ready=read(folder/'READY.json')
        for name,h in ready['files'].items():assert sha(folder/name)==h,(folder,name)
        rt=teacher(i);s=np.asarray(rt.scale);scales.append(dict(name=r['name'],act_scale=s,each_robot_isotropic=bool(np.all(s[:,0]==s[:,1])),all_dimensions_isotropic=bool(np.all(s==s.flat[0])),max_min_ratio=float(s.max()/s.min()),rotation_exact=bool(np.all(s[:,0]==s[:,1]))))
        evidence.append(dict(teacher=r['name'],data=str(folder),ready_sha256=sha(folder/'READY.json'),reuse='train/dev local and teacher observations and train-only normalization reused byte-for-byte; position from own global observation[:2]',gap='new frame-consistent target/noise student and its execution labels',rollouts_generated=0))
    write(HERE/'ACT_SCALE_AUDIT.json',scales);write(HERE/'HISTORY_AUDIT.json',dict(utc=utc(),evidence=evidence,local_reuse_indices=[8,10,12,13],old_global_results='superseded; original files preserved',teacher_supplement_reuse=str(OLD/'teacher_DB_ROWS.json')))
    (HERE/'HISTORY_EVIDENCE.md').write_text('# Historical evidence\n\nRound30 TEACHERS/config/checkpoints and READY files verified by SHA256. All state arrays, observations and normalization reused without reconstructing states or generating rollouts. All 4 valid local students, frozen choices and existing labels are reused; only the ten global students lack frame-consistent checkpoints. Teacher supplementary arms are already in round30/DB and will be read only. Noise gate reuses 1008/1008 with all original non-wrapper imported source hashes unchanged.\n')
    source_inventory('imports_prepare.json')
    print('PREPARE PASS',flush=True)
if __name__=='__main__':main()
