"""Outcome-frozen P1 state/checkpoint inventory and exact DB profile planning."""
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
from diagnostics.controller_q_diversity_v1 import experiment as old
from shared_rollout_db.src.rollout_db import connect, transaction, uid, canonical


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, obj):
    path = HERE / path
    if path.exists():
        assert json.loads(path.read_text()) == obj, f'frozen file would change: {path}'
    else:
        path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    oldidx = ROOT / 'diagnostics/ring_state_selection_transfer_v1/FLOW_CHECKPOINT_INDEX.json'
    source = json.loads((ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json').read_text())
    devproto = json.loads((ROOT/'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/protocol.json').read_text())
    frozen = np.load(ROOT/'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/frozen_predictions.npz')
    y = np.load(ROOT/'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/fresh_rollouts.npz')
    assert len(devproto['states']) == 256
    native_eta = frozen['physical23_eta_n512_logits'].mean(0).argmax(-1)
    discovery = y['outcomes'][:,:,:8,0].mean(-1)
    gain = discovery[np.arange(256), native_eta]-discovery[:,13]
    selected = sorted(range(256), key=lambda i:(-float(gain[i]),devproto['states'][i]['state_uid']))[:12]
    remaining = sorted((i for i in range(256) if i not in selected), key=lambda i:(float(discovery[i,13]),devproto['states'][i]['state_uid']))
    rng = np.random.default_rng(2026101205)
    other=[]
    for chunk in np.array_split(np.asarray(remaining),4):
        other.extend(map(int,rng.choice(chunk,3,replace=False)))
    ids=selected+other
    assert len(ids)==len(set(ids))==24 and all(gain[i]>0 for i in selected)
    states=[]
    for k,i in enumerate(ids):
        s=copy.deepcopy(devproto['states'][i]);s['scenario']='ring_exchange';s['study_split']='p1_development';s['study_index']=k
        states.append(s)
    dump('P1_STATE_SPLIT.json',dict(indices=ids,state_uids=[s['state_uid'] for s in states],adaptive_indices=selected,other_indices=other,
                                    adaptive_discovery_gain=[float(gain[i]) for i in selected],
                                    historical_fixed13_rates=[float(discovery[i,13]) for i in ids],
                                    frozen_q512_sha256=sha(ROOT/'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/frozen_predictions.npz'),
                                    old_outcomes_sha256=sha(ROOT/'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/fresh_rollouts.npz'),
                                    status='FROZEN BEFORE P1 NEW OUTCOMES; all DEVELOPMENT'))
    native=source['controller']
    native_profile=copy.deepcopy(native)
    family=[];profiles={};audit=[]
    baseline_manifest=json.loads((ROOT/'diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json').read_text())
    for row in json.loads(oldidx.read_text()):
        cp=Path(row['path']);name=cp.parent.name
        with cp.open('rb') as f: payload=pickle.load(f)
        cfg=payload['config'];fingerprint=cfg.get('environment_fingerprint')
        frame={
          'b2c23c9e2d0a5ed8638a90651214867dabe93986f8e0dbe1620b1e4e18adbbad':'local',
          '9a3b417c1f542eddb728877e3677a332be93df8771a0bca10fbfb2c8051c8665':'global'
        }.get(fingerprint)
        likely=ROOT/'diagnostics/ring_exchange_stage1'/name.replace('_macflow','_dataset')/'manifest.json'
        fallback=ROOT/'diagnostics/ring_exchange_stage1'/('base_u_v10_local_dataset' if frame=='local' else 'base_u_dataset')/'manifest.json'
        manifest=likely if likely.exists() else fallback
        reasons=[]
        if frame is None:reasons.append('unknown environment fingerprint')
        if not manifest.exists():reasons.append('no compatible physical manifest')
        m=json.loads(manifest.read_text()) if manifest.exists() else {}
        if m.get('environment_fingerprint')!=fingerprint:reasons.append('manifest/checkpoint fingerprint mismatch')
        physical={k:v for k,v in m.get('scenario_config',{}).items() if k!='schema'}
        native_physical={k:v for k,v in baseline_manifest['scenario_config'].items() if k!='schema'}
        if physical!=native_physical:reasons.append('physical geometry/dynamics differ')
        if cfg.get('flow_steps')!=10 or cfg.get('joint_obs_dim')!=92 or cfg.get('joint_act_dim')!=8 or cfg.get('num_agents')!=4 or not cfg.get('normalize'):
            reasons.append('network shape or normalization incompatible')
        if sha(cp)!=row['sha256']:reasons.append('checkpoint SHA changed')
        audit.append(dict(name=name,checkpoint=str(cp),checkpoint_sha256=sha(cp),config_path=row['config'],
                          manifest=str(manifest),manifest_sha256=sha(manifest) if manifest.exists() else None,
                          checkpoint_dataset_manifest_sha256=cfg.get('dataset_manifest_sha256'),
                          selected_manifest_exact_training_hash=manifest.exists() and cfg.get('dataset_manifest_sha256')==sha(manifest),
                          frame=frame,environment_fingerprint=fingerprint,compatible=not reasons,reasons=reasons,
                          jax_enable_x64=False))
        if reasons:continue
        ctrl=copy.deepcopy(native_profile['payload'])
        ctrl.update(base_controller='real_ring_flow_checkpoint_free_t0_ff_round3_v1',flow_checkpoint_sha256=sha(cp),
                    flow_checkpoint_path=str(cp),checkpoint_training_manifest_sha256=cfg.get('dataset_manifest_sha256'),
                    adapter_manifest_path=str(manifest),adapter_manifest_sha256=sha(manifest),
                    observation_frame=frame,checkpoint_environment_fingerprint=fingerprint,
                    execution_callable=str(HERE/'flow_adapter.py'),execution_callable_sha256=sha(HERE/'flow_adapter.py'),
                    controller_source_sha256=sha(HERE/'flow_adapter.py'),
                    conditioning='free true-t0 native Ring FF; checkpoint-specific observed frame; same physical plant and hard projection',
                    jax_enable_x64=False)
        ctrl.pop('legacy_controller_uid_verified_by_original_function',None)
        profiles[name]=dict(controller_uid=uid('ctl',ctrl),payload=ctrl)
        family.append(name)
    dump('P1_CHECKPOINT_COMPATIBILITY.json',dict(rows=audit,compatible_count=len(family),candidate_count=len(audit)))
    assert len(family)<=14
    protocol=dict(schema='controller_q_round3_real_flow_p1_v1',experiment_uid='exp_controller_q_transfer_round3_p1_v1',
                  states=states,controllers={'ring_exchange':profiles},controller_order=family,eta=source['eta'],eta_indices=list(range(16)),
                  seeds=list(range(4)),environments={'ring_exchange':source['environment']},
                  native_v10_controller_uid=native['controller_uid'],source_protocol_sha256=sha(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json'),
                  state_split_sha256=sha(HERE/'P1_STATE_SPLIT.json'),compatibility_sha256=sha(HERE/'P1_CHECKPOINT_COMPATIBILITY.json'),
                  adapter_sha256=sha(HERE/'flow_adapter.py'),
                  ff_operator_sha256=sha(Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004/poc/field_controller.py')),
                  jax_enable_x64=False,horizon=700,dt=.05,placement='FF',
                  scope='DEVELOPMENT pilot, old256 opened; Q4 crossfit not B15')
    dump('P1_PROTOCOL.json',protocol)
    with connect() as db, transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (protocol['experiment_uid'],'Real Flow controller transfer Round3 P1',str(HERE),sha(HERE/'P1_PROTOCOL.json'),sha(__file__),canonical(dict(max_new_rollouts=25000,max_cpu_workers=24))))
        for ctl in profiles.values():
            v=ctl['payload'];existing=db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?',(ctl['controller_uid'],)).fetchone()
            assert existing is None or existing['config_json']==canonical(v)
            db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (ctl['controller_uid'],v['scenario_uid'],v['flow_checkpoint_sha256'],v['orthoflow3_sha256'],uid('safe',dict(code=v['safety_projection_sha256'],cbf=v['cbf'])),str(v['horizon']),str(v['dt']),v['success_semantics'],v['conditioning'],canonical(v['rng']),canonical(v),'EXACT_PROFILE'))
    print(json.dumps(dict(compatible=len(family),names=family,states=len(states),gross_requests=len(family)*24*16*4)))


if __name__=='__main__':main()
