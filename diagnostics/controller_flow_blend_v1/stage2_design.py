"""Called only after the frozen Stage 1 gate passes."""
from common import *
import copy,subprocess
from collections import defaultdict
import numpy as np
from shared_rollout_db.src.rollout_db import connect,transaction,canonical
from diagnostics.controller_q_diversity_v1.experiment import case
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate

ARMS=['R0','R3','R4_action','R_oracle','R5_behavior']
def create():
    assert read(HERE/'STAGE_1_GATE.json')['passed'],'Stage1 gate failed; no Stage2'
    # Freeze separate stage document, never rewrite Stage1 or the draft.
    dest=HERE/'PREREGISTRATION_STAGE_2.md'
    if not dest.exists():
        text=(HERE/'PREREGISTRATION_STAGE_2_DRAFT.md').read_text()
        text=text.replace('（草案；pilot未通过前不执行、不盖阶段收据）','（Stage1门通过后正式冻结）')
        dest.write_text(text)
    receipt(dest)
    folder=HERE/'stage_2';p1=read(HERE/'stage_1/protocol.json');old=read(PREVIOUS/'protocol.json')
    native=read(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    train=copy.deepcopy(old['states'][:96]);test=generate('blend_test',32,2026170000,native);probe=generate('blend_probe',16,2026180000,native)
    with connect(True) as db:
        collisions=[]
        for s in test+probe:
            z=db.execute('SELECT state_uid FROM state WHERE state_uid=? OR content_hash=?',(s['state_uid'],s['content_hash'])).fetchall()
            # Resume already-registered study states only if exact index proven by frozen protocol.
            if z and not (folder/'protocol.json').exists():collisions.append(s['state_uid'])
        count=db.execute('SELECT count(*) FROM state').fetchone()[0]
    assert not collisions
    assert len({s['content_hash'] for s in train+test+probe})==144
    # Also cover unindexed diagnostics/PoC protocols: physical content only,
    # independent of aliases, metadata and rng_namespace.
    paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore','diagnostics'],cwd=ROOT,text=True).splitlines()
    physical_hash=lambda s:__import__('hashlib').sha256(canonical(s['physical']).encode()).hexdigest()
    new_hashes={physical_hash(s) for s in test+probe};historical_hashes=set();checked=[]
    for text in paths:
        path=ROOT/text
        if not path.name.lower().endswith('protocol.json') or HERE in path.parents:continue
        q=read(path)
        if not isinstance(q,dict):continue
        ss=q.get('states',[]);ss=list(ss.values()) if isinstance(ss,dict) else ss
        pool=q.get('pool',[]);pool=list(pool.values()) if isinstance(pool,dict) else pool
        ss=ss if isinstance(ss,list) else []
        pool=pool if isinstance(pool,list) else []
        ss=ss+[v.get('state',v) for v in pool if isinstance(v,dict)]
        for s in ss:
            if isinstance(s,dict) and 'physical' in s:historical_hashes.add(physical_hash(s))
        checked.append(dict(path=str(path),sha256=sha(path)))
    assert not new_hashes&historical_hashes,'Historical physical-state overlap'
    audit=dict(historical_db_states=count,new_test_count=32,new_probe_count=16,collisions=collisions,draw_seed_test=2026170000,draw_seed_probe=2026180000,
        reused_train_states=96,train_outcomes_opened=True,new_test_outcomes_opened=False,
        historical_protocols=checked,historical_physical_hash_count=len(historical_hashes),historical_physical_hash_collisions=0)
    if not (folder/'STATE_DISJOINT_RECEIPT.json').exists():freeze(folder/'STATE_DISJOINT_RECEIPT.json',audit)
    p=copy.deepcopy(p1);p.update(schema='controller_flow_blend_stage2_v1',experiment_uid='exp_controller_flow_blend_v1_stage_2',states=train+test,probe_states=probe,
        eta_indices=list(range(16)),preregistration_sha256=sha(dest),stage1_gate_sha256=sha(HERE/'STAGE_1_GATE.json'),train_states=96,test_states=32,probe_states_count=16,fit_seeds=[17,23,41],arms=ARMS)
    freeze(folder/'protocol.json',p)
    masks=read(PREVIOUS/'TRAIN_ETA_MASK.json');freeze(folder/'TRAIN_ETA_MASK.json',{s['state_uid']:masks[s['state_uid']] for s in train})
    assert [sum(j in masks[s['state_uid']] for s in train) for j in range(16)]==[48]*16
    cases=[];groups=defaultdict(list)
    for n in p['controller_order']:
        for s in train+test:
            for j in masks[s['state_uid']] if s in train else range(16):
                for seed in range(4):
                    r=case(p,s,n,j,seed);cases.append(r);groups[r['state_uid'],r['eta_uid'],r['controller_uid']].append(r['seed_key'])
    assert len(cases)==61440
    cases.sort(key=lambda r:(r['controller_name'],r['split'],r['state_index'],r['eta_index'],r['seed']))
    freeze(folder/'CASES.json',cases);freeze(folder/'PLANNED_ROLLOUTS.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=v) for (s,e,c),v in groups.items()]))
    # Training never reads test labels. Validation uses original Flow only.
    tasks=[]
    for split,folds in [('LOPO',range(4)),('LOAO',range(2))]:
        for fold in folds:
            held=[n for n in p['controller_order'] if (p['controllers']['ring_exchange'][n]['payload']['blend']['pair']==fold if split=='LOPO' else p['controllers']['ring_exchange'][n]['payload']['blend']['pair']//2==fold)]
            for coverage in ('DENSE','SPARSE'):
                src=p['anchor_order']+([n for n in p['controller_order'] if n not in held] if coverage=='DENSE' else [])
                for arm in ARMS:tasks.append(dict(split=split,fold=fold,coverage=coverage,arm=arm,sources=src,held=held))
    freeze(folder/'FIT_TASKS.json',tasks)
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (p['experiment_uid'],'Round5 coverage x representation',str(folder),sha(folder/'protocol.json'),sha(__file__),canonical(dict(new_cap=65000))))
        for s in test+probe:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),canonical(p['environments']['ring_exchange']['config']),canonical(dict(metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],p['experiment_uid']))
    print(json.dumps(dict(requests=len(cases),fit_tasks=len(tasks),train=96,test=32,probe=16)))

def inventory():
    folder=HERE/'stage_2';pre=read(folder/'CACHE_PREFLIGHT.json')['summary'];cases=read(folder/'CASES.json');missing=[];unknown=[];cached=[]
    with connect(True) as db:
        for r in cases:
            z=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if z is None:missing.append(r)
            else:
                assert not z['conflict_quarantined'];assert all(z[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                (unknown if z['numerical_failure'] else cached).append(r['rollout_uid'])
    assert pre['ambiguous']==0 and pre['genuinely_missing']==len(missing)+len(unknown)
    assert len(missing)+832<=65000
    freeze(folder/'MISSING_CASES.json',missing);freeze(folder/'CACHE_INVENTORY.json',dict(total=len(cases),missing=len(missing),cached_valid=len(cached),cached_unknown=len(unknown),quarantined=0,new_cap=65000,r5_extra_short_requests=832,preflight_summary=pre))
    print(json.dumps(read(folder/'CACHE_INVENTORY.json')))

if __name__=='__main__':globals()[sys.argv[1]]()
