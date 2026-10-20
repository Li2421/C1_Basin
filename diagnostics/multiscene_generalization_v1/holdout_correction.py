"""Correct a cohort construction defect without changing any frozen policy."""
from common import *
from catalog import proposal
from design_generated import identities
from registry import register

def prepare():
    original=read(HERE/'CATALOG_FROZEN.json');cfg=read(HERE/'HOLDOUT_CORRECTION_CONFIG.json')
    catalog={};rows=[];registered=[];overlap={}
    for ordinal,(family,states) in enumerate(original['catalog'].items()):
        training=[s for s in states if s['role']=='train'];evaluation=[s for s in states if s['role']=='evaluation']
        train_seeds={s['geometry_seed'] for s in training};overlap[family]=sorted(train_seeds&{s['geometry_seed'] for s in evaluation})
        corrected=[s for s in states if s['role']!='evaluation']
        hashes={hashlib.sha256(json.dumps(s['rectangles'],sort_keys=True).encode()).hexdigest() for s in training}
        for map_index in range(8):
            base=cfg['geometry_base']+ordinal*cfg['family_offset']+map_index*cfg['map_stride']
            for retry in range(10000):
                gs=base+retry
                try:
                    group=[]
                    for index in range(map_index*4,map_index*4+4):
                        ts=cfg['task_base']+ordinal*1000000+index*100000
                        for tr in range(10000):
                            try:s=proposal(family,'evaluation',index,gs,ts+tr);break
                            except ValueError:pass
                        else:raise ValueError('no valid task at this geometry')
                        group.append(s)
                    assert gs not in train_seeds
                    assert hashlib.sha256(json.dumps(group[0]['rectangles'],sort_keys=True).encode()).hexdigest() not in hashes
                    break
                except ValueError:continue
            else:raise RuntimeError('holdout seed range exhausted '+family)
            corrected+=group
            for s in group:
                c,r=identities(s,'sequential_expert','expert');rows.append(c);registered+=r
        catalog[family]=corrected
    write(HERE/'ORIGINAL_GEOMETRY_OVERLAP_AUDIT.json',dict(overlap=overlap,
        original_navigation_cohort_status='INVALID_FOR_CAPABILITY: training/evaluation geometry overlap',
        old_labels_retained=True,old_labels_not_rerun=True,frozen_policies_unchanged=True))
    write(HERE/'HOLDOUT_VALIDATED_CATALOG.json',catalog)
    register('holdout_joint_0',rows,registered)

def accept(stage):
    catalog=read(HERE/'HOLDOUT_VALIDATED_CATALOG.json');failed=[]
    for c in read(HERE/f'{stage}_CASES.json'):
        s=c['state'];row=next(v for v in catalog[s['family']] if v['alias']==s['alias'])
        raw=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json")
        row['joint_proof']=dict(raw_path=str(HERE/stage/'raw'/f"{c['rollout_uid']}.json"),trace_sha256=raw['trace_sha256'],success=bool(raw['success']),outcome=raw['outcome'])
        if raw['success']:row['joint_feasibility']='certified'
        else:failed.append(row)
    write(HERE/'HOLDOUT_VALIDATED_CATALOG.json',catalog)
    return failed

def retry(number,failed):
    catalog=read(HERE/'HOLDOUT_VALIDATED_CATALOG.json');rows=[];registered=[]
    for family,old_seed in sorted({(s['family'],s['geometry_seed']) for s in failed}):
        oldgroup=[s for s in catalog[family] if s['geometry_seed']==old_seed and s['role']=='evaluation']
        gs=old_seed+1
        for _ in range(10000):
            try:newgroup=[proposal(s['family'],s['role'],s['index'],gs,s['task_seed']+1) for s in oldgroup];break
            except ValueError:gs+=1
        else:raise RuntimeError('next holdout geometry limit exhausted '+family)
        for old,new in zip(oldgroup,newgroup):
            i=next(i for i,s in enumerate(catalog[family]) if s['alias']==old['alias']);catalog[family][i]=new
            c,r=identities(new,'sequential_expert','expert');rows.append(c);registered+=r
    write(HERE/'HOLDOUT_VALIDATED_CATALOG.json',catalog)
    register('holdout_joint_'+str(number),rows,registered)

def finalize():
    catalog=read(HERE/'HOLDOUT_VALIDATED_CATALOG.json');checks={}
    for f,states in catalog.items():
        t=[s for s in states if s['role']=='train'];e=[s for s in states if s['role']=='evaluation']
        th={hashlib.sha256(json.dumps(s['rectangles'],sort_keys=True).encode()).hexdigest() for s in t};eh={hashlib.sha256(json.dumps(s['rectangles'],sort_keys=True).encode()).hexdigest() for s in e}
        assert len({s['geometry_seed'] for s in t})==16 and len({s['geometry_seed'] for s in e})==8
        assert not ({s['geometry_seed'] for s in t}&{s['geometry_seed'] for s in e}) and not th&eh
        assert all(s['joint_feasibility']=='certified' for s in states)
        checks[f]=dict(training_geometries=16,evaluation_geometries=8,seed_intersection=[],geometry_content_intersection=[],joint_valid_states=len(states))
    write(HERE/'CATALOG_FINAL.json',dict(families=list(catalog),catalog=catalog,gaps=[],holdout_checks=checks))
    seal('CATALOG_FINAL.receipt.json',[HERE/'CATALOG_FINAL.json',HERE/'HOLDOUT_CORRECTION_CONFIG.json',HERE/'ORIGINAL_GEOMETRY_OVERLAP_AUDIT.json'],before_valid_navigation_gate=True)

if __name__=='__main__':globals()[sys.argv[1]]()
