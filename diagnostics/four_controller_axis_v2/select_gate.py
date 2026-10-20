"""Freeze outcome-stratified historical G1a records before replay."""
from __future__ import annotations
import datetime, hashlib, json, math, sqlite3, struct
from pathlib import Path
from shared_rollout_db.src.rollout_db import canonical

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUNS=('base_u_v12_world_timeout_goal_recovery_canonical_source_balanced_macflow',
      'base_u_v10_world_timeout_goal_recovery_source_balanced_wide_macflow')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bits(v):return struct.pack('>d',float(v)).hex()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def main():
    receipt=json.loads((HERE/'PREREGISTRATION.receipt.json').read_text())
    assert receipt['preregistration_sha256']==sha(HERE/'PREREGISTRATION.md')
    assert not (HERE/'G1A_RESULT.json').exists()
    db=sqlite3.connect(f'file:{ROOT/"shared_rollout_db/rollout.sqlite"}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    chosen=[];counts={}
    for run in RUNS:
        ck=ROOT/'diagnostics/four_way_intersection_stage1'/run/'best.pkl';h=sha(ck)
        for outcome in ('success','timeout'):
            rows=db.execute('''SELECT r.rollout_uid,r.state_uid,r.eta_uid,r.controller_uid,r.seed_key,
                r.outcome,r.episode_length,r.min_agent_distance,r.min_wall_distance,r.numerical_failure,
                r.original_source_file,r.raw_record_hash,r.experiment_uid
                FROM rollout r JOIN controller_config c ON r.controller_uid=c.controller_uid
                WHERE c.flow_checkpoint_sha256=? AND c.conditioning_version=? AND r.outcome=?
                AND r.numerical_failure=0 AND r.conflict_quarantined=0
                ORDER BY r.rollout_uid''',(h,'training_dev_intermediate_state_episode_eta_v1+committed_base_t0_then_alt_future_v1',outcome)).fetchall()
            n=len(rows);assert n>=8,(run,outcome,n);counts[run+'|'+outcome]=n
            positions=[math.floor((j+.5)*n/8) for j in range(8)]
            assert len(set(positions))==8
            for rank in positions:
                r=rows[rank];p=Path(r['original_source_file']);assert p.is_file()
                envelope=json.loads(p.read_text().splitlines()[0]);raw=envelope['record']
                assert sha(p) and hashlib.sha256(canonical(raw).encode()).hexdigest()==r['raw_record_hash']
                assert raw['state_uid']==r['state_uid'] and raw['controller_uid']==r['controller_uid']
                assert raw['outcome']==r['outcome'] and raw['episode_length']==int(r['episode_length'])
                assert bits(raw['minimum_agent_clearance'])==bits(r['min_agent_distance'])
                assert bits(raw['minimum_wall_clearance'])==bits(r['min_wall_distance'])
                assert json.loads(r['seed_key'])['future_index']==raw['future_index']
                assert raw['future_root_seed']==2026093011 and raw['rng_namespace'] is not None
                assert raw['controller_chain']=='orthoflow3' and raw['flow_checkpoint_sha256']==h
                chosen.append(dict(run=run,checkpoint_path=str(ck),checkpoint_sha256=h,
                    outcome_stratum=outcome,rank=rank,stratum_count=n,rollout_uid=r['rollout_uid'],
                    experiment_uid=r['experiment_uid'],source_path=str(p),source_sha256=sha(p),
                    state_uid=r['state_uid'],eta_uid=r['eta_uid'],controller_uid=r['controller_uid'],seed_key=r['seed_key'],
                    expected=dict(outcome=r['outcome'],episode_length=int(r['episode_length']),
                        min_agent_distance_float64_be=bits(r['min_agent_distance']),
                        min_wall_distance_float64_be=bits(r['min_wall_distance'])),
                    original=dict(state_id=raw['state_id'],initial_state_index=raw['initial_state_index'],
                        future_index=raw['future_index'],future_root_seed=raw['future_root_seed'],
                        rng_namespace=raw['rng_namespace'],initial_positions=raw['initial_positions'],
                        initial_velocities=raw['initial_velocities'],goals=raw['goals'],eta=raw['eta'])))
    assert len(chosen)==32 and len({r['rollout_uid'] for r in chosen})==32
    selection=dict(status='FROZEN_BEFORE_G0_G1A_G2_REPLAY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        preregistration_sha256=sha(HERE/'PREREGISTRATION.md'),selection_code_sha256=sha(__file__),
        rule='per checkpoint/outcome sort rollout_uid, select floor((j+0.5)*N/8) j=0..7',
        counts=counts,records=chosen,g1b_rollout_uids=[next(r['rollout_uid'] for r in chosen if r['run']==run and r['outcome_stratum']==outcome)
            for run in RUNS for outcome in ('success','timeout')])
    write(HERE/'G1A_SELECTION.json',selection)
    write(HERE/'G1A_SELECTION.receipt.json',dict(status=selection['status'],utc=selection['utc'],
        selection_sha256=sha(HERE/'G1A_SELECTION.json'),preregistration_sha256=selection['preregistration_sha256']))
    print(json.dumps(dict(status=selection['status'],counts=counts,total=len(chosen))))

if __name__=='__main__':main()
