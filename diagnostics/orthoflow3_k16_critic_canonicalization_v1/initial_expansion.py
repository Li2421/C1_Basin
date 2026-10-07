"""Outcome-blind initial-state coverage for Phase A, inherited parent splits.

The audited v2 new-scenario states are all mid-trajectory. Select 32 train and
8 dev original parents/scenario by fixed hash, never by failure/outcome.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a

MAIN_OUT=a.OUT
EXTRA=MAIN_OUT/'initial_expansion'
ORIGINAL_STATES=a.states


def selected_raw(sc):
    original=ORIGINAL_STATES()
    parents={r['source_initial_state_id']:r for r in original if r['scenario']==sc}
    rows=a.load(a.ROOT/'datasets/orthoflow3_basin_dataset_v1/work'/sc/'parent_manifest.json')['states']
    out=[]
    for split,n in [('train',32),('validation',8)]:
        eligible=[r for r in rows if r['uid'] in parents and parents[r['uid']]['split']==split]
        eligible.sort(key=lambda r:a.digest(['phase-A-initial-coverage-v1',sc,r['uid']]))
        for row in eligible[:n]:
            assert row['provenance']['opened_test_archives']==0
            assert row['rollout_id']==parents[row['uid']]['parent_episode_id']
            out.append(row)
    assert len(out)==40
    return out


def runtimes(scenario=None):
    a.bd.OUT=EXTRA/'phase_a/evidence';a.bd.WORK=a.bd.OUT
    for sc in ('four_way_intersection','ring_exchange'):
        if scenario and scenario!=sc:continue
        rt=a.bd.TrainingRuntime(sc,selected_raw(sc),parent=True)
        a.register(rt)
        yield sc,rt


def create_states():
    extra=[];source={r['source_initial_state_id']:r for r in ORIGINAL_STATES() if r['scenario']!='double_bottleneck'}
    # The generic register function reads protocol.json; no outcomes are read.
    a.dump('protocol.json',{'phase':'A','rule':'hash-selected 32train+8dev parent initial states per new scenario','frozen_test':False})
    for sc,rt in runtimes():
        for state in rt.states:
            flat,descriptor=a.frozen.FrozenNewRuntime.conditioning(rt,state)
            env=rt.make_env();rt.reset(env,state)
            original=source[state['uid']]
            extra.append({'state_id':state['alias'],'state_uid':state['uid'],'scenario':sc,
                          'split':original['split'],'parent_episode_id':original['parent_episode_id'],
                          'source_initial_state_id':state['uid'],'source_rollout_type':'legitimate_parent_initial_state',
                          'timestep':0,'conditioning':{'flat':flat,'native_observation':rt.observation(env).tolist(),
                                                       'reference_current_raw_flow':np.asarray(flat[-8:]).reshape(4,2).tolist()},
                          'structured_state':{'positions':env.positions.tolist(),'velocities':env.velocities.tolist(),'goals':env.goals.tolist(),'timestep':0},
                          'environment_descriptor':descriptor,'controller_uid':rt.controllers['orthoflow3']['uid'],
                          'zero_sufficient':None,'provenance':state['provenance']})
    a.dump('states.json',extra)
    a.dump('acquisition_manifest.json',{'reason':'v2 contains no t=0 Four/Ring conditioning states',
           'diagnosis_source':'v2 timestep metadata only; no frozen-test outcomes',
           'rule':'first32train+8dev parents by SHA256(phase-A-initial-coverage-v1,scenario,parent_uid)',
           'parent_split_inherited':True,'eta_source':'frozen generator,mean+16stochastic,noextra',
           'full_Q16':True,'new_states':len(extra),'counts':dict(Counter(r['scenario']+':'+r['split'] for r in extra))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','preflight','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    a.OUT=EXTRA
    if args.stage=='freeze':create_states()
    a.states=lambda:a.load(EXTRA/'states.json')
    a.runtimes=runtimes
    value=a.freeze() if args.stage=='freeze' else a.cache_preflight() if args.stage=='preflight' else a.run(args.shard,args.shards) if args.stage=='run' else a.collect()
    print(a.canonical(value),flush=True)
