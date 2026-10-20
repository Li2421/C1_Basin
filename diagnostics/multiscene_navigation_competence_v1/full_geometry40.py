"""Unexecuted full-stage geometry preparation; both pilot gates are mandatory."""
from common import *
from catalog import proposal
from design_generated import identities
from registry import register
from shared_rollout_db.src.rollout_db import connect

FAMILIES=tuple('ACDEG')
def require_pilot():
    for f in 'BF':
        p=HERE/f'NAVIGATION_CAPABILITY_{f}.json'
        assert p.exists() and read(p)['family_passed'],'Both pilot families must pass before the full stage'

def verify_full():
    receipt=read(HERE/'FULL_STAGE_IMPLEMENTATION.receipt.json')
    for path,expected in receipt['files'].items():assert sha(path)==expected,('Full-stage implementation differs from freeze',path)

def catalog_path(family,frozen=True):
    assert family in FAMILIES
    return HERE/(f'FULL_{family}_GEOMETRY_'+('FROZEN.json' if frozen else 'CANDIDATES.json'))

def read_catalog(family):return read(catalog_path(family))[family]

def exclusions():
    old=read(HERE/'ROUND39_CATALOG_FROZEN.json')['catalog']
    all_states=[s for states in old.values() for s in states]
    # This is the explicitly authorized read-only previous delivery, not a prohibited prefix.
    final=read(D/'multiscene_generalization_v1/CATALOG_FINAL.json')['catalog']
    all_states += [s for states in final.values() for s in states]
    all_states += [s for states in read(HERE/'PILOT_GEOMETRY_FROZEN.json').values() for s in states]
    for f in FAMILIES:
        for frozen in [True,False]:
            p=catalog_path(f,frozen)
            if p.exists():all_states+=read(p)[f]
    return {s['geometry_seed'] for s in all_states},{json.dumps(s['rectangles']) for s in all_states}

def proposal_in_block(family,role,index,base,first,task_base,used,contents,rejected):
    for offset in range(first,10000):
        gs=base+offset;ts=task_base+offset
        try:
            s=proposal(family,role,index,gs,ts)
            if gs in used or json.dumps(s['rectangles']) in contents:raise ValueError('geometry seed or rectangle content already used')
            s.update(alias=f'round40_full_{family}_{role}_{index}_{gs}_{ts}',origin='round40 full new',seed_block=base,task_seed_block=task_base)
            used.add(gs);contents.add(json.dumps(s['rectangles']));return s
        except ValueError as error:rejected.append(dict(family=family,role=role,index=index,geometry_seed=gs,task_seed=ts,reason=str(error)))
    write(HERE/f'FULL_{family}_{role}_{index}_GEOMETRY_EXHAUSTED.json',dict(family=family,role=role,index=index,geometry_seed_block=base,task_seed_block=task_base,first_offset=first,limit=10000,rejections=rejected))
    raise RuntimeError('Preregistered geometry seed block exhausted')

def prepare(family):
    require_pilot();verify_full();assert family in FAMILIES
    assert not catalog_path(family,False).exists()
    config=read(HERE/'FULL_STAGE_GEOMETRY_SEEDS_FROZEN.json');task_config=read(HERE/'FULL_STAGE_JOINT_TASK_SEEDS_FROZEN.json')
    old=read(HERE/'ROUND39_CATALOG_FROZEN.json')['catalog'][family]
    states=[dict(s,origin='round39 training reused') for s in old if s['role']=='train'];assert len(states)==16
    history=[]
    for s in states:
        assert s['joint_feasibility']=='certified' and s['joint_proof']['success']
        assert Path(s['joint_proof']['raw_path']).resolve().is_relative_to(D/'multiscene_generalization_v1')
        proof=read(s['joint_proof']['raw_path']);assert proof['success'] and proof['trace_sha256']==s['joint_proof']['trace_sha256']
        assert Path(proof['trace_path']).resolve().is_relative_to(D/'multiscene_generalization_v1')
        assert sha(proof['trace_path'])==proof['trace_sha256']
        with connect(True) as db:row=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(proof['controller_uid'],)).fetchone()
        assert row is not None;profile=json.loads(row['config_json'])
        assert profile['plant_sha256']==sha(ROOT/'bottleneck_family/environment.py'),'Historical joint proof native plant incompatible'
        assert all(profile['source_hashes'][name]==sha(HERE/name) for name in ['geometry.py','sequential_expert.py']),'Historical joint proof generated geometry or sequential reference incompatible'
        history.append(dict(geometry_seed=s['geometry_seed'],raw_path=s['joint_proof']['raw_path'],trace_path=proof['trace_path'],trace_sha256=proof['trace_sha256'],controller_uid=proof['controller_uid'],native_source_compatible=True,reused_without_rollout=True))
    write(HERE/f'FULL_{family}_HISTORICAL_EVIDENCE.json',dict(old_training_joint_proofs=history,previous_navigation_delivery=str(D/'multiscene_generalization_v1/ALL_VALID_NAVIGATION_GATES.json'),old_navigation_checkpoints_compatible=False,old_navigation_reason='35-dimensional old policies and pure-pursuit expert differ from fixed39-dimensional route observations and new expert',missing_evidence=['56 new geometry joint-feasibility proofs','2048 fixed expert gate episodes','fixed39-dimensional demonstrations and three relabel rounds','four final policy checkpoints and16 fresh navigation episodes each'],reproduction_gate_reused='REPRODUCTION_GATE.json',no_passed_gate_reruns=True))
    used,contents=exclusions();rejected=[];cases=[];rows=[];ordinal='ABCDEFG'.index(family)
    for role,count,base,task_offset in [('train_new',48,config['remaining_training_base'],0),('navigation_holdout',8,config['remaining_navigation_holdout_base'],task_config['navigation_offset'])]:
        for i in range(count):
            block=base+ordinal*config['family_offset']+i*config['map_stride'];tb=task_config['base']+ordinal*task_config['family_offset']+task_offset+i*task_config['map_stride']
            s=proposal_in_block(family,role,i,block,0,tb,used,contents,rejected);states.append(s)
            c,r=identities(s,'sequential_expert','expert');cases.append(c);rows+=r
    write(catalog_path(family,False),{family:states});write(HERE/f'FULL_{family}_GEOMETRY_REJECTIONS.json',rejected)
    stage=f'full_joint_{family}_0';register(stage,cases,rows)
    seal(f'FULL_{family}_GEOMETRY_PROPOSALS.receipt.json',[catalog_path(family,False),HERE/f'FULL_{family}_GEOMETRY_REJECTIONS.json',HERE/'FULL_STAGE_GEOMETRY_SEEDS_FROZEN.json',HERE/'FULL_STAGE_JOINT_TASK_SEEDS_FROZEN.json',Path(__file__)],old_training_joint_proofs_reused=16,before_new_joint_proofs=True)
    return stage

def accept(family,stage,attempt):
    require_pilot();verify_full();catalog=read(catalog_path(family,False));states=catalog[family];cases=read(HERE/f'{stage}_CASES.json');used,contents=exclusions();retry=[];rows=[];rejected=[]
    for c in cases:
        s=c['state'];i=next(i for i,q in enumerate(states) if q['alias']==s['alias']);r=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json")
        if r['success']:
            states[i].update(joint_feasibility='certified',joint_proof=dict(raw_path=str(HERE/stage/'raw'/f"{c['rollout_uid']}.json"),trace_sha256=r['trace_sha256'],success=True,outcome=r['outcome']))
        else:
            first=s['geometry_seed']-s['seed_block']+1
            q=proposal_in_block(family,s['role'],s['index'],s['seed_block'],first,s['task_seed_block'],used,contents,rejected)
            states[i]=q;nc,nr=identities(q,'sequential_expert','expert');retry.append(nc);rows+=nr
    write(catalog_path(family,False),catalog);write(HERE/f'FULL_{family}_JOINT_REJECTIONS_{attempt}.json',rejected)
    if retry:
        next_stage=f'full_joint_{family}_{attempt}';register(next_stage,retry,rows);return next_stage
    assert len(states)==72 and all(s['joint_feasibility']=='certified' for s in states)
    train=[s for s in states if s['role'] in ['train','train_new']];held=[s for s in states if s['role']=='navigation_holdout'];assert len(train)==64 and len(held)==8
    assert not ({s['geometry_seed'] for s in train}&{s['geometry_seed'] for s in held})
    assert not ({json.dumps(s['rectangles']) for s in train}&{json.dumps(s['rectangles']) for s in held})
    write(catalog_path(family),catalog);seal(f'FULL_{family}_GEOMETRY_FROZEN.receipt.json',[catalog_path(family)],training_geometries=64,new_navigation_holdout_geometries=8,old_joint_proofs_reused=16,new_joint_proofs=56,unknowns_not_rerun=True)
    return None
