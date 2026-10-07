#!/usr/bin/env python3
"""Freeze current DB generator means and plan Q16; NO rollout execution."""
import json
import sqlite3
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

ROOT=Path('/home/zhihan/research/Basin_C1')
OUT=Path(__file__).resolve().parent
GEN=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
HARD=ROOT/'diagnostics/orthoflow3_db_generator_necessity_v1'
BASE=ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1'
sys.path.insert(0,str(ROOT))
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.train_generator import Generator,mean_eta
from shared_rollout_db.src.rollout_db import eta_identity

model=Generator()
template=model.init(jax.random.PRNGKey(0),jnp.zeros((1,80),jnp.float32),'DB')
selected=json.loads((GEN/'generator_training.json').read_text())['selected']
params=serialization.from_bytes(template,Path(selected['checkpoint']).read_bytes())
norm=json.loads((BASE/'dataset_manifest.json').read_text())['state_normalization']['DB']
mu=np.asarray(norm['mean'],np.float32);sd=np.asarray(norm['std'],np.float32)
states=json.loads((HARD/'hard_state_manifest.json').read_text())['states']
con=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
controller='ctl_ceef00016f37f243923d0c7568f7b017db6854059447fd2daaa7d797b1d5258b'
seed_keys=[json.dumps({'future_index':i},sort_keys=True,separators=(',',':')) for i in range(16)]
frozen=[];requests=[]
for s in states:
    h=np.asarray(json.loads((HARD/'raw'/f"conditioning_{s['state_id']}.json").read_text())['h_feature'],np.float32)
    if h.shape!=(80,):raise RuntimeError(f'Incompatible DB h: {s["state_id"]} {h.shape}')
    eta=np.asarray(mean_eta(model.apply(params,jnp.asarray(((h-mu)/sd)[None]),'DB')))[0].astype(float).tolist()
    row=con.execute('''SELECT s.state_uid FROM state_alias a JOIN state s USING(state_uid)
         WHERE a.alias=? AND s.identity_quality='CONDITIONING_EXACT' ''',(s['state_id'],)).fetchall()
    if len(row)!=1:raise RuntimeError(f'Conditioning-exact state ambiguous: {s["state_id"]}')
    uid=row[0][0]
    available={r[0] for r in con.execute('''SELECT DISTINCT r.controller_uid FROM rollout r
        JOIN state s USING(state_uid) WHERE s.source_group=?''',(s['source_group'],))}
    if controller not in available:raise RuntimeError(f'Frozen hard-panel controller unavailable: {s["state_id"]}')
    frozen.append({'state_id':s['state_id'],'state_uid':uid,'source_group':s['source_group'],'eta':eta})
    requests.append({'state_uid':uid,'eta_uid':eta_identity(eta)[0],
                     'controller_uid':controller,'seed_keys':seed_keys})
(OUT/'db_hard_frozen_generator_mean.json').write_text(json.dumps({'generator_checkpoint_sha256':selected['sha256'],
    'hard_panel_controller_uid':controller,
    'states':frozen,'test_outcomes_used':False},indent=2,sort_keys=True)+'\n')
(OUT/'db_hard_mean_planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2,sort_keys=True)+'\n')
print(json.dumps({'states':len(states),'requested_continuations':len(requests)*16,'rollouts_executed':0}))
