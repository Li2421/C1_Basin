"""Read-only historical evidence and byte-audited Round18 training plan."""
from core import *
import pickle

def main():
    prereg()
    historic=read(R15/'HISTORICAL_CODE_AUDIT.json')
    assert historic['passed'] and len(historic['files'])==18
    for item in historic['files']:
        assert sha(ROOT/item['file'])==item['sha256'],item['file']
    old=read(R15/'TRAINING_PLAN.json')
    original={x['run']:x for x in old if x['wave']==1}
    assert len(original)==14
    assert {'base_u_v3_macflow','base_u_plus_w_v3_macflow'}<=set(original)
    v3=['base_u_v3_macflow','base_u_plus_w_v3_macflow']
    controls=sorted(set(original)-set(v3))
    assert len(controls)==12
    plan=[]
    def add(run,arch,offset,split,family):
        o=original[run];manifest=ROOT/o['dataset']/'manifest.json'
        assert sha(manifest)==o['dataset_manifest_sha256']
        m=read(manifest);assert m['complete']
        cfg=read(ROOT/o['original_config'])
        prior=pickle.loads((ROOT/o['original_best']).read_bytes())
        assert cfg['steps']==o['steps'] and prior['config']['environment_fingerprint']==m['environment_fingerprint']
        name=f'{run}__arch{arch[0]}x{len(arch)}__plus{offset}'
        plan.append(dict(name=name,run=run,root_recipe=run,split=split,family=family,
                         original_seed=o['original_seed'],seed=o['original_seed']+offset,seed_offset=offset,
                         actor_hidden_dims=list(arch),steps=o['steps'],batch_size=256,
                         dataset=o['dataset'],dataset_manifest_sha256=o['dataset_manifest_sha256'],
                         config=o['original_config'],original_best=o['original_best']))
    for run in v3:
        for arch in ((256,256,256),(384,384,384),(512,512,512,512)):
            for offset in range(50001,50005):
                add(run,arch,offset,'TRAIN-new' if offset<=50002 else 'TEST','v3')
    for j,run in enumerate(controls,1):
        add(run,(384,384,384),50001,'TRAIN-new' if j%2 else 'TEST','control')
    assert len(plan)==36 and len({x['name'] for x in plan})==36
    assert sum(x['split']=='TRAIN-new' for x in plan)==18
    assert sum(x['split']=='TEST' for x in plan)==18
    write(HERE/'TRAINING_PLAN.json',plan)
    write(HERE/'HISTORICAL_EVIDENCE_AUDIT.json',dict(status='PASS',utc=utc(),
         historical_training_package_files_verified=18,
         historical_code_audit_path=str(R15/'HISTORICAL_CODE_AUDIT.json'),
         historical_code_audit_sha256=sha(R15/'HISTORICAL_CODE_AUDIT.json'),
         round15_labels_sha256=sha(R15/'dataset.jsonl'),
         round15_R5_sha256=sha(R15/'R5_FEATURES.npz'),
         round17_labels_sha256=sha(R17/'LABELS.npz'),
         round17_R5_sha256=sha(R17/'R5_FEATURES.npz'),
         round16_model_freeze_sha256=sha(R16/'MODEL_FREEZE.receipt.json'),
         reused=['Round15 and Round17 labels with unchanged execution semantics',
                 'Round15/Round17 R5 probes', 'Round16 frozen A3 checkpoints and transform for C0'],
         missing=['36 new Flow checkpoints','36 new-controller R5 probes',
                  '109 controller eta2 probes','32 TRAIN-new and 48 TEST fresh states',
                  'TRAIN-new and TEST Q4 rollout labels'],
         plan_sha256=sha(HERE/'TRAINING_PLAN.json'),
         test_split_names=[x['name'] for x in plan if x['split']=='TEST'],
         train_split_names=[x['name'] for x in plan if x['split']=='TRAIN-new'],
         video_exception='Pure Q data generation/training/evaluation'))
    print(json.dumps(dict(status='PASS',controllers=len(plan),train=18,test=18,controls=controls)))

if __name__=='__main__':main()
