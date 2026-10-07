#!/usr/bin/env python3
"""Freeze fresh-seed Q64 requests for already selected DB policies."""
import json
from pathlib import Path
from shared_rollout_db.src.rollout_db import eta_identity

ROOT = Path('/home/zhihan/research/Basin_C1')
HERE = Path(__file__).resolve().parent
PRIOR = ROOT/'diagnostics/orthoflow3_db_mode_free_mustdo_v1'
AUDIT = ROOT/'diagnostics/orthoflow3_pipeline_dataset_evidence_audit_v1'
mean = {r['state_id']: r for r in json.loads((AUDIT/'db_hard_frozen_generator_mean.json').read_text())['states']}
prop = json.loads((PRIOR/'db_frozen_proposals.json').read_text())
controller = json.loads((AUDIT/'db_hard_frozen_generator_mean.json').read_text())['hard_panel_controller_uid']
rows = []
requests = []
for s in prop['states']:
    sid = s['state_id']
    selected = s['critic_choice_K16']
    for method, eta in [('generator_mean',mean[sid]['eta']),('critic_K16',s['eta'][selected])]:
        eid = eta_identity(eta)[0]
        rows.append({'state_id':sid,'state_uid':s['state_uid'],'source_group':s['source_group'],
                     'method':method,'eta':eta,'eta_uid':eid,'critic_sample_index':selected if method=='critic_K16' else None})
        requests.append({'state_uid':s['state_uid'],'eta_uid':eid,'controller_uid':controller,
                         'seed_keys':[json.dumps({'future_index':i},sort_keys=True,separators=(',',':')) for i in range(16,64)]})
assert len(rows)==96 and len({r['source_group'] for r in rows})==48
HERE.mkdir(parents=True,exist_ok=True)
(HERE/'frozen_q64_manifest.json').write_text(json.dumps({'states':rows,'controller_uid':controller,
    'future_indices':list(range(16,64)),'matched_seeds':True,
    'critic_scores_and_choices_frozen_before_Q16_outcomes':prop['selection_before_proposal_outcomes'],
    'generator_sha256':prop['generator_sha256'],'critic_checkpoints':prop['critic_checkpoints']},indent=2)+'\n')
(HERE/'planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
assets=json.loads((ROOT/'diagnostics/orthoflow3_db_generator_necessity_v1/frozen_assets.json').read_text())
fixed=assets['transformed_eta'][assets['best_fixed_mode']]
baseline_requests=[]
for s in prop['states']:
    for eta in ([0.0,0.0,0.0],fixed):
        baseline_requests.append({'state_uid':s['state_uid'],'eta_uid':eta_identity(eta)[0],
            'controller_uid':controller,'seed_keys':[json.dumps({'future_index':i},sort_keys=True,separators=(',',':')) for i in range(64)]})
(HERE/'baseline_cache_audit_plan.json').write_text(json.dumps({'requests':baseline_requests},indent=2)+'\n')
print(json.dumps({'policies':len(rows),'requested':len(rows)*48}))
