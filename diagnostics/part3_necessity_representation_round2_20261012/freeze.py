"""Freeze Round-2 outcome-blind confirmation protocol from P1 discovery only."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
P1=ROOT/'diagnostics/controller_q_transfer_round3_v1'
from diagnostics.controller_q_transfer_round3_v1.state_generator import generate
from diagnostics.controller_q_diversity_v1.experiment import case
from shared_rollout_db.src.rollout_db import connect,eta_identity,canonical,transaction,uid

PAIR=('base_u_v2_macflow','base_u_v10_local_macflow')
EXP='exp_part3_round2_ring_real_flow_q16_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def freeze(path,obj):
    path=HERE/path
    content=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==content, f'frozen file changed: {path}'
    else:path.write_text(content)

def choose_eta(y,names):
    """P1 seeds 0/1 only; maximize reproducible opposite *global* preference.

    Unknown is zero in this score (conservative success lower bound).
    Ties: higher empirical two-choice V, then smallest eta indices.
    """
    q=y[[names.index(name) for name in PAIR],:,:,:2].mean(-1)
    rows=[]
    for a in range(16):
        for b in range(a+1,16):
            diffs=(q[:,:,a]-q[:,:,b]).mean(axis=1)
            score=max(0.,min(float(diffs[0]),-float(diffs[1])),min(-float(diffs[0]),float(diffs[1])))
            v=(np.maximum(q[0,:,a],q[0,:,b])+np.maximum(q[1,:,a],q[1,:,b]))/2
            v-=np.maximum((q[0,:,a]+q[1,:,a])/2,(q[0,:,b]+q[1,:,b])/2)
            rows.append(dict(indices=[a,b],opposite_preference_score=score,V=float(v.mean()),
                             pair_mean_preference_differences=diffs.tolist()))
    return sorted(rows,key=lambda r:(-r['opposite_preference_score'],-r['V'],r['indices']))

def main():
    p=json.loads((P1/'P1_PROTOCOL.json').read_text())
    with np.load(P1/'P1_LABELS.npz') as z:
        names=z['controller_names'].tolist();ranked=choose_eta(z['Y'],names)
    winner=ranked[0];assert winner['indices']==[1,2],winner
    states=generate('part3_round2_confirm',5,2026101200,
                    {'environment':p['environments']['ring_exchange']})
    for i,s in enumerate(states):
        s['study_split']='round2_confirmation'
        s['study_index']=i
    assert len({s['state_uid'] for s in states})==5
    # Identity checks only. Do not inspect any new outcome or replace a state
    # based on controller success/failure.
    with connect(True) as db:
        for s in states:
            if not (HERE/'protocol.json').exists():
                assert db.execute('SELECT COUNT(*) FROM state WHERE state_uid=? OR content_hash=?',
                                  (s['state_uid'],s['content_hash'])).fetchone()[0]==0,s['state_uid']
            else:
                row=db.execute('SELECT content_hash,physical_state_json FROM state WHERE state_uid=?',(s['state_uid'],)).fetchone()
                assert row and row['content_hash']==s['content_hash'] and row['physical_state_json']==canonical(s['physical'])
    controls={name:p['controllers']['ring_exchange'][name] for name in PAIR}
    for ctl in controls.values():
        profile=ctl['payload']
        assert sha(profile['flow_checkpoint_path'])==profile['flow_checkpoint_sha256']
        assert sha(profile['adapter_manifest_path'])==profile['adapter_manifest_sha256']
    assert sha(P1/'flow_adapter.py')==p['adapter_sha256']
    indices=winner['indices']
    protocol=dict(schema='part3_round2_real_flow_q16_confirmation_v1',experiment_uid=EXP,
        source_p1_protocol_sha256=sha(P1/'P1_PROTOCOL.json'),
        source_p1_discovery_labels_sha256=sha(P1/'P1_LABELS.npz'),
        discovery_seed_indices=[0,1],discovery_choice=winner,
        discovery_ranking=ranked,controller_order=list(PAIR),
        controllers={'ring_exchange':controls},states=states,
        state_generation=dict(generator='state_generator.generate',split='part3_round2_confirm',
            count=5,seed_start=2026101200,seed_end=2026101204,
            selection='first five sequential draws; all absent from global state UID and content hash before execution',
            no_outcome_selection=True),
        eta_indices=indices,eta=[p['eta'][j] for j in indices],seeds=list(range(16)),
        seed_rule='existing key_for: PRNGKey(2026100403), fold SHA256(state uid), seed, physical step; common both controllers/eta',
        environments=p['environments'],placement='FF',horizon=700,dt=.05,
        adapter_sha256=p['adapter_sha256'],ff_operator_sha256=p['ff_operator_sha256'],
        jax_enable_x64=False,max_new_continuations=320,
        outcome_rule="collision_free_success and termination=='success' and no numerical exception",
        analysis_plan='PREREGISTRATION.md')
    freeze('protocol.json',protocol)
    cases=[]
    # old.case indexes eta in p['eta']; preserve original K16 integer identity.
    for name in PAIR:
        for s in states:
            for j in indices:
                for seed in range(16):cases.append(case(p | {'states':states},s,name,j,seed))
    assert len(cases)==320 and len({c['rollout_uid'] for c in cases})==320
    freeze('cases.json',cases)
    groups={}
    for c in cases:
        key=(c['state_uid'],c['eta_uid'],c['controller_uid'])
        groups.setdefault(key,[]).append(c['seed_key'])
    freeze('planned_rollouts.json',dict(requests=[dict(state_uid=s,eta_uid=e,controller_uid=c,seed_keys=keys)
                                              for (s,e,c),keys in groups.items()]))
    # Registration supplies exact identities for cache preflight and journal
    # ingestion. It contains no labels and cannot affect the frozen design.
    with connect() as db,transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXP,'C1 Part3 Round2 real Flow necessity Q16',str(HERE),sha(HERE/'protocol.json'),sha(__file__),canonical(dict(max_new_continuations=320))))
        for s in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                       (s['state_uid'],s['scenario_uid'],s['family'],s['content_hash'],canonical(s['physical']),
                        canonical(p['environments']['ring_exchange']['config']),canonical(dict(source_alias=s['uid'],metadata=s['metadata'])),'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',(s['scenario_uid'],s['uid'],s['state_uid'],EXP))
        for j in indices:
            eid,v,hx=eta_identity(p['eta'][j])
            db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)',(eid,*v,hx))
        for ctl in controls.values():
            v=ctl['payload'];prior=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
            assert prior is not None and prior[0]==canonical(v)
    print(json.dumps(dict(frozen=True,eta_indices=indices,state_uids=[s['state_uid'] for s in states],
                          cases=len(cases),protocol_sha256=sha(HERE/'protocol.json'))))

if __name__=='__main__':main()
