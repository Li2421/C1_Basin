"""Freeze targeted local TRAIN/VAL/TEST states before new VAL/TEST outcomes."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path[:0]=[str(SYSROOT),str(ROOT)]
import numpy as np
from shared_rollout_db.src.rollout_db import canonical,uid,eta_identity,connect
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.prepare_cache_plan import SCENARIO,CORRECTION_CONFIG,RNG
sys.path.insert(0,str(SYSROOT))
PRIOR=ROOT/'diagnostics/orthoflow3_toy_critic_local_state_probe_v1'
OLD=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
EXPERIMENT='exp_orthoflow3_toy_critic_local_supervision_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvwrite(p,rs):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def key(state,kind,eta):
    h=hashlib.sha256(canonical({'initial_positions':state['initial_positions']}).encode()).hexdigest()
    return {'episode_index':state['episode_index'],'kind':kind,
            'state_uid':uid('state',{'scenario':SCENARIO,'content':h}),
            'eta_uid':eta_identity(eta)[0],'controller_uid':uid('ctl',CORRECTION_CONFIG)}
def main():
    if (OUT/'frozen_protocol.json').exists():raise RuntimeError('Already frozen')
    assert read(PRIOR/'cache_postflight.json')['summary']['exact_reusable']==320
    prior=read(PRIOR/'frozen_proposals.json')
    root=read(OLD/'frozen_proposals.json')
    parent={s['episode_index']:s for s in prior['references']}
    assert len(parent)==5
    runner=OLD/'run_rollouts.py'
    protocol={'experiment_uid':EXPERIMENT,'freeze_time':datetime.now(timezone.utc).isoformat(),
      'objective':'distinguish local missing-data support from model fitting on five targeted critic-error families',
      'population_scope':'within-family local interpolation/extrapolation diagnostic only; no population generalization claim',
      'parent_episode_indices':sorted(parent),
      'TRAIN':'original center and +/-0.010m positions, two original eta per state, all cached Q16',
      'VAL':'agent0 initial x +/-0.005m, same two eta, frozen and evaluated before checkpoint selection',
      'TEST':'agent0 initial x +/-0.015m, same two eta; outcomes are acquired only after checkpoints are frozen',
      'physical_axis':'agent0 x only; all other initial values, parent rollout_id and continuation seed policy frozen',
      'VAL_expected_new_continuations':320,'TEST_expected_new_continuations':320,
      'total_expected_new_continuations':640,'seed_keys':list(range(16)),
      'controller_config':CORRECTION_CONFIG,'rng':RNG,'new_eta':0,
      'train_local_pairs':30,'val_pairs':20,'test_pairs':20,
      'selection':'VAL pair-equal soft-label NLL, earlier step on ties; TEST untouched until checkpoint freeze',
      'model':'same frozen W1 Toy critic architecture and pure-NLL objective, three seeds; compare no-local replay and local-label replay from identical pretrained checkpoints',
      'optimization':'AdamW 1e-3 weight decay1e-4, gradient clip5; original 64 structured +64 wide pairs plus32 extra per step; local variant extras are local pairs, control extras are original pairs; identical generic draws',
      'evaluation_caveat':'same five source families across partitions, so tests local held-out perturbations, not new source groups; original 200-state hard set was already used for diagnosis',
      'source_group':'all perturbations retain parent source group',
      'runner_sha256':sha(runner),'prior_frozen_sha256':sha(PRIOR/'frozen_proposals.json'),
      'critic_architecture_source_sha256':sha(ROOT/'diagnostics/orthoflow3_ranking_aware_critic_v1/run_experiment.py')}
    OUT.mkdir(exist_ok=True)
    (OUT/'logs').mkdir(exist_ok=True)
    write(OUT/'frozen_protocol.json',protocol)
    from single_integrator.environment import Config,GiveWayEnv
    wide=read(ROOT/'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json')
    env=GiveWayEnv(Config(**wide['environment']))
    from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.freeze_proposals import features_for_cohort,critic_scores
    phases={'VAL':(-.005,.005),'TEST':(-.015,.015)}
    manifest={'TRAIN':{'parent_center_and_prior_perturbations_sha256':sha(PRIOR/'frozen_proposals.json'),'source_groups':sorted({p['source_group'] for p in parent.values()})}}
    collision_uids=[]
    with connect(True) as con:
        for phase, offsets in phases.items():
            dst=OUT/phase.lower();dst.mkdir(exist_ok=True)
            (dst/'logs').mkdir(exist_ok=True)
            states=[]
            for ep in sorted(parent):
                p=parent[ep]
                for offset in offsets:
                    pos=np.asarray(p['initial_positions'],np.float64).copy();pos[0,0]+=offset
                    env.reset(pos)
                    i=len(states)
                    state={'episode_index':i,'parent_episode_index':ep,'state_id':f'local_supervision_{phase.lower()}_p{ep:04d}_{i}',
                      'source_group':p['source_group'],'rollout_id':p['rollout_id'],
                      'initial_positions':pos.tolist(),'physical_offset_m':offset,
                      'eta':p['eta'],'bad_proposal_index':p['bad_proposal_index'],'good_proposal_index':p['good_proposal_index']}
                    states.append(state)
            h=features_for_cohort(states,wide['environment'],wide['cbf'])
            eta=np.asarray([[s['eta'][k] for k in ('frozen_bad','frozen_good')] for s in states],np.float64)
            scores,critics=critic_scores(h,eta)
            frozen_scores=[]
            for i,s in enumerate(states):
                s['h_raw']=h[i].astype(float).tolist()
                s['h_sha256']=hashlib.sha256(h[i].astype(np.float64).tobytes()).hexdigest()
                for j,kind in enumerate(('frozen_bad','frozen_good')):
                    frozen_scores.append({'episode_index':i,'parent_episode_index':s['parent_episode_index'],
                        'physical_offset_m':s['physical_offset_m'],'kind':kind,'frozen_critic_logit':float(scores[i,j])})
            write(dst/'frozen_proposals.json',{'phase':phase,'states':states,'critic_checkpoints':critics,
                 'protocol_sha256':sha(OUT/'frozen_protocol.json'),'before_phase_outcomes':True})
            np.savez_compressed(dst/'cohort_features.npz',h_raw=h)
            mapping=[key(s,k,e) for s in states for k,e in s['eta'].items()]
            csvwrite(dst/'candidate_cache_keys.csv',mapping)
            csvwrite(dst/'frozen_critic_scores.csv',frozen_scores)
            requests=[{k:r[k] for k in ('state_uid','eta_uid','controller_uid')} |
                      {'seed_keys':[canonical({'future_index':j}) for j in range(16)]} for r in mapping]
            write(dst/'planned_rollouts.json',{'requests':requests})
            for r in mapping:
                if con.execute('SELECT 1 FROM state WHERE state_uid=?',(r['state_uid'],)).fetchone():collision_uids.append(r['state_uid'])
            manifest[phase]={'state_count':len(states),'pair_count':len(mapping),'requested_continuations':len(requests)*16,
                'source_groups':sorted({s['source_group'] for s in states}),
                'h_rms_to_parent_max':None,'frozen_proposals_sha256':sha(dst/'frozen_proposals.json')}
    if collision_uids:raise RuntimeError(('VAL/TEST state key already in DB',collision_uids))
    manifest['new_state_uids_already_present']=collision_uids
    assert manifest['TRAIN']['source_groups']==manifest['VAL']['source_groups']==manifest['TEST']['source_groups']
    write(OUT/'split_manifest.json',manifest)
    with connect() as con:
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,start_time,metadata_json) VALUES(?,?,?,?,?,?)',
                    (EXPERIMENT,OUT.name,str(OUT),sha(OUT/'frozen_protocol.json'),protocol['freeze_time'],canonical(protocol)))
    write(OUT/'working_state.json',{'stage':'FROZEN_AWAITING_PREFLIGHT','requested_VAL':320,'requested_TEST':320,'new_rollout':0})
    print(json.dumps({'train_local_pairs':30,'val_requested':320,'test_requested':320,'new_state_keys_already_present':len(collision_uids)}))
if __name__=='__main__':main()
