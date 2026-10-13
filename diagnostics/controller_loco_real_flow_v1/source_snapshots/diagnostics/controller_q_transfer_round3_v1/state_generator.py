"""Outcome-blind natural Ring state and historical physical23 feature helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np
from ring_exchange.environment import LocalFrameConfig, sample_instance
from ring_exchange.local_frame import vectors_to_local
from shared_rollout_db.src.rollout_db import uid

ROOT=Path(__file__).resolve().parents[2]


def generate(split, count, seed_start, native_protocol):
    cfg=LocalFrameConfig(**native_protocol['environment']['config']['config'])
    scenario_uid=native_protocol['environment']['scenario_uid']
    rows=[]
    for i in range(count):
        seed=seed_start+i
        instance=sample_instance(seed,'development',cfg)
        physical=dict(positions=instance.positions.tolist(),velocities=instance.velocities.tolist(),
                      goals=instance.goals.tolist(),timestep=0)
        alias=f'controller_q_transfer_round3_v1_{split}_{seed}'
        content=uid('content',dict(physical=physical,reset='empty monitor true-t0'))
        row=dict(uid=alias,state_uid=uid('state',dict(scenario=scenario_uid,content=content)),
                 physical=physical,scenario='ring_exchange',scenario_uid=scenario_uid,
                 content_hash=content,rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),
                 family=alias,metadata=dict(split=split,draw_seed=seed,outcome_blind=True,source='native Ring development generator'),
                 study_split=split,study_index=i)
        rows.append(row)
    assert len({x['state_uid'] for x in rows})==count
    return rows


def physical23(states, normalizer_path=None):
    if normalizer_path is None:
        normalizer_path=ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/physical_inputs.npz'
    z=np.load(normalizer_path)
    wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi
    raw=[]
    for s in states:
        p=s['physical'];pos=np.asarray(p['positions']);v=np.asarray(p['velocities']);goal=np.asarray(p['goals'])
        theta=np.arctan2(pos[:,1],pos[:,0]);gt=np.arctan2(goal[:,1],goal[:,0])
        rp=np.linalg.norm(pos,axis=1);rg=np.linalg.norm(goal,axis=1)
        rel=wrap(theta[1:]-theta[0]-np.arange(1,4)*np.pi/2);gd=wrap(gt-theta-np.pi);vl=vectors_to_local(v,pos)
        raw.append(np.r_[rp,rel,rg,gd,vl.ravel()])
    raw=np.asarray(raw,np.float32)
    return (raw-z['physical_center'])/z['physical_scale']
