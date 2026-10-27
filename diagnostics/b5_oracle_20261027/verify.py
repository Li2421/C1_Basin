"""Validate identities and evidence; preserve raw journals and emit a canonical export."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src import rollout_db as db,cache_writer
from shared_control.correction_b5 import FrozenYield,YieldConfig
import numpy as np

def main():
    # Recover only this experiment's missing registration metadata from its
    # preflight manifests. Never rerun a completed physical trajectory.
    repair=[]
    manifests=list(OUT.glob('*_planned_rollouts.json'))
    cases=[]
    for p in manifests:
        cases.extend((c,str(p))for c in json.load(p.open()).get('exact_configs',[]))
    cases.extend((c,str(OUT/'LEGACY_FROZEN.json'))for c in json.load(open(OUT/'LEGACY_FROZEN.json'))['tasks'])
    with db.connect()as con,db.transaction(con):
        for c,path in cases:
            if con.execute('SELECT 1 FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone():continue
            physical=c.get('physical')or{k:c['state'][k]for k in ('positions','velocities','goals')}
            con.execute('INSERT INTO state(state_uid,scenario_uid,content_hash,physical_state_json,provenance_json,identity_quality)VALUES(?,?,?,?,?,?)',
                (c['state_uid'],c['scenario_uid'],db.uid('physical',physical),db.canonical(physical),db.canonical(dict(source_manifest=path,repair='required provenance_json restored')),'CONTENT_EXACT'))
            repair.append(c['state_uid'])
        con.execute('UPDATE experiment SET path=?,metadata_json=? WHERE experiment_uid=?',
            (str(OUT/'legacy_traces'),db.canonical(dict(legacy_checkpoint_backtest=True,registration_repaired=True)),'exp_b5_legacy_20261027'))
    repair_path=OUT/'cache_registration_repair.json'
    if repair or not repair_path.exists():repair_path.write_text(json.dumps(dict(repaired_state_uids=repair,
        reason='Initial INSERT OR IGNORE omitted required provenance_json, so state metadata was silently skipped. Preflight ran but first merger classified these journals as ambiguous. Metadata now reconstructed from original manifests, with readback assertions; original raw journals preserved; no rerollout.'),indent=2)+'\n')
    rows=list(map(json.loads,(OUT/'candidates.jsonl').open()))
    legacy=[]
    for name in ('legacy_candidates.jsonl','legacy_prior_candidates.jsonl'):legacy.extend(map(json.loads,(OUT/name).open()))
    assert len({r['rollout_uid']for r in rows+legacy})==len(rows+legacy)
    amended=[];canonical=[];missing=[];traces=set();physics={};configs={}
    with db.connect(True)as con:
        for raw in rows+legacy:
            r=dict(raw);sid=r['state_uid'];cid=r['controller_uid']
            if sid not in physics:
                x=con.execute('SELECT * FROM state WHERE state_uid=?',(sid,)).fetchone();assert x
                physics[sid]=(json.loads(x['physical_state_json']),x['scenario_uid'])
            if cid not in configs:
                x=con.execute('SELECT * FROM controller_config WHERE controller_uid=?',(cid,)).fetchone();assert x
                configs[cid]=(json.loads(x['config_json']),x['flow_checkpoint_sha256'])
            physical,scid=physics[sid];payload,cp=configs[cid];eta=r['spec']['eta'][:3]
            assert db.eta_identity(eta)[0]==r['eta_uid']
            if r.get('eta')!=eta:amended.append(dict(rollout_uid=r['rollout_uid'],field='eta',old=r.get('eta'),new=eta,reason='Add missing historical eta metadata.'if r.get('eta')is None else 'Correct reused source eta metadata; spec eta and registered UID were already correct.'))
            r.update(eta=eta,scenario_uid=scid,seed_key=db.canonical(dict(future_index=r['seed'])),future_index=r['seed'],
                initial_positions=physical['positions'],initial_velocities=physical['velocities'],goals=physical['goals'],
                flow_checkpoint_sha256=cp,cache_identity_schema='registered_exact_uid_v1',deadlock=r.get('deadlock',0))
            assert r['rollout_uid']==db.uid('roll',dict(state=sid,eta=r['eta_uid'],controller=cid,seed=r['seed_key']))
            assert not (r['success']and(r.get('collision')or r.get('numerical_failure')))
            if r.get('trace'):assert Path(r['trace']).exists();traces.add(r['trace'])
            role=r['spec'].get('role')
            if role:
                scenario=con.execute('SELECT metadata_json FROM scenario WHERE scenario_uid=?',(scid,)).fetchone()
                config=json.loads(scenario['metadata_json']);config=config.get('config',config)
                p=np.asarray(physical['positions']);g=np.asarray(physical['goals'])
                if isinstance(role,str):
                    d=np.linalg.norm(p[:,None]-p[None,:],axis=-1);np.fill_diagonal(d,np.inf);i,j=map(int,np.unravel_index(np.argmin(d),d.shape))
                    rc=YieldConfig(j,i)if role=='reverse_nearest_hold'else YieldConfig(i,j,'back',2.)
                else:rc=YieldConfig(**role)
                length=2*config['agent_radius']+config.get('agent_collision_margin',0.)+payload.get('cbf',{}).get('separation_buffer',1e-4)
                frozen=FrozenYield(rc,p,g,length)
                r['frozen_role_geometry']=dict(role=rc.__dict__,waypoint=frozen.waypoint.tolist(),forward=frozen.forward.tolist(),length_scale=length)
            canonical.append(r)
            q=con.execute('SELECT success,collision,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if q is None:missing.append(r)
            else:
                assert q['success']==r['success']and q['collision']==r.get('collision',0)and q['numerical_failure']==r.get('numerical_failure',0)and not q['conflict_quarantined']
    # Preserve original raw files. The export adds explicit waypoint coordinates,
    # canonical eta metadata, physical provenance, and resolved reuse identities.
    with (OUT/'candidates_verified.jsonl').open('w')as f:
        for r in canonical:f.write(json.dumps(r,allow_nan=False)+'\n')
    (OUT/'candidate_metadata_amendments.json').write_text(json.dumps(amended,indent=2)+'\n')
    if missing:
        for experiment,subset in [('exp_b5_oracle_20261027',[r for r in missing if 'scene'in r]),('exp_b5_legacy_20261027',[r for r in missing if 'scene'not in r])]:
            if not subset:continue
            cache_writer.append_journal(subset,experiment,worker_id='verified_evidence_export')
            subprocess.run([sys.executable,'-m','shared_rollout_db.src.merger','--experiment',experiment],cwd=ROOT,check=True)
    with db.connect(True)as con:
        for r in canonical:
            q=con.execute('SELECT success,conflict_quarantined FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            assert q and q['success']==r['success']and not q['conflict_quarantined']
    files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in OUT.glob('*.py')}
    basis=ROOT/'shared_control/correction_b5.py'
    files['shared_control/correction_b5.py']=dict(bytes=basis.stat().st_size,sha256=hashlib.sha256(basis.read_bytes()).hexdigest())
    result=dict(passed=True,rows=len(canonical),unique_saved_traces=len(traces),registered_exact_cache_rows=len(canonical),
        imported_compatible_evidence=len(missing),metadata_amendments=len(amended),duplicate_ids=0,conflicts=0,
        code_files=files,reference_package_checks='NOT_RUN: reference ZIP absent',
        data_backup='Raw JSONL, NPZ and SQLite remain local ignored research evidence; Git commit is not their backup.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
