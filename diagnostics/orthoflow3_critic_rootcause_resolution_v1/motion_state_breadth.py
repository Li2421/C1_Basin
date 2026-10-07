"""Finish an existing source matrix, preserving all frozen tests and semantics."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from . import function_support as old
from . import seed_replication as engine
from shared_rollout_db.src.rollout_db import connect,transaction,canonical
from shared_rollout_db.src.planner import preflight

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
OUT=ROOT/'motion_state_breadth'
RULE=ROOT/'motion_state_breadth_protocol.json'
EXP='exp_orthoflow3_motion_state_breadth160_v1'
read,write,sha=old.read,old.write,old.sha
NAMES=old.OLD+old.NAMES


def configure():
    engine.OUT=OUT;engine.EXP=EXP;engine.SHARDS=18
    engine.parent.SHARDS=18;engine.parent.CONTROLLERS=NAMES
    engine.configure()


def prepare():
    assert not (OUT/'protocol.json').exists(),'Preserve frozen design'
    OUT.mkdir(exist_ok=True)
    source_states=read(old.SOURCE/'states.json');source_pairs=read(old.SOURCE/'pairs.json')
    data=np.load(old.SOURCE/'dataset.npz')
    all_train=[s for i,s in enumerate(source_states) if data['source'][2*i]!='old46' and s['split']=='train']
    existing=read(old.OUT/'states.json');used={s['uid'] for s in existing}
    remaining=[s for s in all_train if s['uid'] not in used]
    key=lambda s:hashlib.sha256(('c1_controller_function_support_v1\0'+s['source_group']).encode()).hexdigest()
    remaining=sorted(remaining,key=key)
    assert len(all_train)==160 and len(remaining)==96
    assert not {s['source_group'] for s in remaining}&{s['source_group'] for s in existing}
    # State manifests only, never outcomes, establish exclusion from confirmations.
    forbidden=set()
    for folder in ROOT.glob('*confirmation*'):
        path=folder/'states.json'
        if not path.exists():continue
        payload=read(path)
        if isinstance(payload,list):
            forbidden.update(s.get('source_group') for s in payload if isinstance(s,dict))
    assert not {s['source_group'] for s in remaining}&forbidden
    states=[];pairs=[];indices=[]
    for i,initial in enumerate(remaining):
        state=copy.deepcopy(initial);indices.append(state['index']);state['index']=state['repair_index']=i
        states.append(state)
        pp=[p for p in source_pairs if p['state_uid']==state['uid']]
        assert len(pp)==2
        for original in pp:
            p=copy.deepcopy(original);p.update(state_index=i,dataset_state_index=i,target_seeds=4)
            assert p['split']=='train';pairs.append(p)
    physical=read(old.SOURCE/'physical.json')
    write(OUT/'states.json',states);write(OUT/'pairs.json',pairs)
    write(OUT/'physical.json',[physical[j] for j in indices])
    entities=np.load(old.SOURCE/'frozen_model_entities.npz')
    np.savez_compressed(OUT/'frozen_model_entities.npz',**{k:entities[k][indices] for k in entities.files})
    protocol=copy.deepcopy(read(old.OUT/'protocol.json'))
    protocol.update(experiment_uid=EXP,experiment='motion_state_breadth160',source_train_families=96,
        source_validation_families=0,requested_seed_count=9216,rule_sha256=sha(RULE),new_TEST_families=0,
        source_rule='All remaining96membersoforiginal160TRAIN,oldoutcomeblindhashordering;noTESTorVALlabelselection',
        target_labels_used=False,generator_modified=False)
    write(OUT/'protocol.json',protocol)
    requests=[];byname={}
    for profile in protocol['profiles']:
        assert sha(profile['path'])==profile['sha256']
        part=[dict(state_uid=p['state_uid'],eta_uid=p['eta_uid'],controller_uid=profile['controller_uid'],
            seed_keys=[canonical({'future_index':i}) for i in range(4)]) for p in pairs]
        write(OUT/f'planned_{profile["name"]}.json',{'requests':part});requests+=part;byname[profile['name']]=part
    write(OUT/'planned_rollouts.json',{'requests':requests})
    for batch,names in enumerate((NAMES[:8],NAMES[8:])):
        write(OUT/f'planned_batch{batch}.json',{'requests':[r for name in names for r in byname[name]]})
    pair_index={(p['state_uid'],p['eta_index']):i for i,p in enumerate(source_pairs)}
    original_pairs=np.array([pair_index[p['state_uid'],p['eta_index']] for p in pairs])
    for ci,name in enumerate(old.OLD):
        np.savez_compressed(OUT/f'inputs_{name}.npz',**{k:data[k][ci,original_pairs] for k in ('context','agent_response','valid')})
    with connect() as db,transaction(db):
        db.execute('INSERT INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXP,'motion_state_breadth160',str(OUT),sha(OUT/'protocol.json'),sha(__file__),canonical({'TRAIN_only':True,'target_labels_used':False})))
    write(OUT/'working_state.json',dict(phase='prepared',new_rollouts=0,rule_sha256=sha(RULE)))
    # Run the mandatory public CLI, not an experiment-local substitute.
    for stem in ('rollouts',*NAMES,'batch0','batch1'):
        manifest=OUT/f'planned_{stem}.json'
        output=OUT/('cache_preflight.json' if stem=='rollouts' else f'cache_preflight_{stem}.json')
        result=subprocess.run([sys.executable,'-m','shared_rollout_db.plan','--manifest',str(manifest),'--output',str(output)],
            cwd=BASE,check=True,capture_output=True,text=True)
    configure();engine.filter_numerical()
    summary=read(OUT/'cache_preflight.json')['summary']
    profiles=protocol['profiles'];identities=[]
    with connect(True) as db:
        for p in pairs:
            assert db.execute('SELECT 1 FROM state WHERE state_uid=?',(p['state_uid'],)).fetchone()
        for p in profiles:
            row=db.execute('SELECT * FROM controller_config WHERE controller_uid=?',(p['controller_uid'],)).fetchone()
            assert row['flow_checkpoint_sha256']==p['sha256'] and row['compatibility_quality']=='EXACT_PROFILE'
            identities.append(p['controller_uid'])
    write(OUT/'identity_audit.json',dict(source_only=True,old_VAL_unchanged=True,families=96,
        independent_confirmation_group_overlap=0,profiles=identities,signatures_relaxed=False,
        missing_is_not_semantic_incompatibility=True,cache_summary=summary))
    print(dict(preflight=summary,numerical_exclusions=read(OUT/'preexisting_numerical_exclusions.json')['count']),flush=True)


def run(batch,worker):
    configure()
    # Only eight newly incomplete functions. Cached first four are never run.
    for name in old.NAMES[4*batch:4*batch+4]:
        engine.run(name,worker)
        import jax,gc
        jax.clear_caches();gc.collect()


def merge(batch):
    configure()
    from diagnostics.orthoflow3_controller_training_repair_v1 import alignment_audit
    previous=alignment_audit.main
    try:
        alignment_audit.main=lambda:None
        engine.native.merge()
    finally:alignment_audit.main=previous
    manifest=OUT/f'planned_batch{batch}.json'
    stats=dict(requested=0,valid=0,numerical=0,missing=0,collision=0,conflict=0)
    with connect(True) as db:
        for r in read(manifest)['requests']:
            rows={a['seed_key']:a for a in db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?',
                (r['state_uid'],r['eta_uid'],r['controller_uid']))}
            for key in r['seed_keys']:
                stats['requested']+=1
                if key not in rows:stats['missing']+=1;continue
                a=rows[key];assert a['compatibility_quality']=='EXACT_REUSE'
                stats['numerical' if a['numerical_failure'] else 'valid']+=1
                stats['collision']+=a['collision'];stats['conflict']+=a['conflict_quarantined']
    assert stats['missing']==stats['conflict']==0
    post=preflight(manifest);assert post['summary']['genuinely_missing']==stats['numerical']
    write(OUT/f'cache_postflight_batch{batch}.json',post);write(OUT/f'completed_batch{batch}.json',stats)
    if batch==1:
        engine.parent.postflight()
    print(dict(batch=batch,stats=stats,postflight=post['summary']),flush=True)


def inputs(index):
    from . import support_inputs
    from . import goal_response as rest
    from . import goal_velocity_response as motion
    name=NAMES[index]
    if index>=4:
        support_inputs.OUT=OUT;support_inputs.features(name)
    rest.SOURCE=OUT;rest.OUT=OUT/'goal_rest';rest.OUT.mkdir(exist_ok=True);rest.RULE=RULE
    rest.build(index)
    # The existing moving query's build requires its own narrow protocol; reuse
    # the measurement itself, never alter a frozen source protocol.
    import time
    from diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_context import RichRuntime
    profile=read(OUT/'protocol.json')['profiles'][index];rt=RichRuntime('ring_exchange',profile['path'])
    pairs=read(OUT/'pairs.json');physical=read(OUT/'physical.json')
    values=[];seconds=[]
    for pair in pairs:
        state=physical[pair['state_index']];assert state['state_uid']==pair['state_uid']
        start=time.perf_counter();values.append(motion.measurement(rt,state['physical'],np.array(pair['eta'],float)))
        seconds.append(time.perf_counter()-start)
    folder=OUT/'goal_motion';folder.mkdir(exist_ok=True)
    assert not (folder/f'inputs_{name}.npz').exists()
    np.savez_compressed(folder/f'inputs_{name}.npz',goal_motion_response=np.asarray(values),valid=np.isfinite(values).all(1),seconds=seconds)
    write(folder/f'audit_{name}.json',dict(controller_sha256=profile['sha256'],code_sha256=sha(__file__),
        no_environment_steps=True,new_task_rollouts=0,target_labels_used=False,physical_sha256=sha(OUT/'physical.json')))
    print(dict(input_controller=name,completed=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','merge','inputs'))
    parser.add_argument('--batch',type=int,default=0);parser.add_argument('--index',type=int,default=0);a=parser.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run(a.batch,a.index)
    elif a.action=='merge':merge(a.batch)
    else:inputs(a.index)
