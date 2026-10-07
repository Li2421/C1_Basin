#!/usr/bin/env python3
"""Read-only original experiment lineage for frozen training pairs."""
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT=Path('/home/zhihan/research/Basin_C1')
OUT=Path(__file__).resolve().parent
BASE=ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1/pair_table.parquet'
STRUCT=ROOT/'diagnostics/orthoflow3_structured_continuous_q_data_v1'

def category(name):
    n=name.lower()
    if 'mode_free_k_sweep' in n or 'mode_free_generator_critic_hard_cohort' in n:return 'hard_cohort_generator_proposals'
    if 'generator' in n or 'deformable' in n:return 'generator_or_local_mode_probes'
    if 'shared_eta_codebook' in n or 'sparse_mode' in n or 'shared_mode_transfer' in n:return 'old_selector_or_codebook_anchors'
    if 'basin' in n or 'geometry' in n or 'intrusion' in n:return 'grid_basin_search_or_boundary'
    if 'recovery' in n or 'perturb' in n:return 'recovery_states'
    if 'fresh' in n or 'random' in n or 'sobol' in n:return 'fresh_random_probes'
    if 'cross_transfer' in n or 'point_learning' in n:return 'historical_target_transfer'
    return 'other_historical'

def main():
    base=pq.read_table(BASE).to_pylist()
    gen=[]
    for r in base:
        if r['state_split']!='train':continue
        if not (0<=r['eta1']<=1.25 and -.5<=r['eta2']<=.5 and 0<=r['eta3']<=.75):continue
        if r['empirical_q']>=.9 or r['empirical_q']<=.5:
            gen.append(('generator_'+r['scenario'],r['state_uid'],r['eta_uid'],r['controller_uid']))
    old=pq.read_table(STRUCT/'structured_pair_table.parquet').to_pylist()
    wide=pq.read_table(STRUCT/'sparse_matched_control.parquet').to_pylist()
    structured=[r for r in old if r['matrix_partition']=='TRAIN_TRAIN']
    sk={(r['state_uid'],r['eta_uid']) for r in structured}
    toyctl=json.loads((ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1/dataset_manifest.json').read_text())['controllers']['Toy']
    critic=[('critic_Toy_structured',r['state_uid'],r['eta_uid'],toyctl) for r in structured]
    critic += [('critic_Toy_wide',r['state_uid'],r['eta_uid'],r['controller_uid']) for r in wide
               if (r['state_uid'],r['eta_uid']) not in sk]
    pairs=list(dict.fromkeys(gen+critic))
    con=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
    count=Counter(); missing=Counter(); names=defaultdict(Counter)
    for role,st,eta,ctl in pairs:
        result=con.execute('''SELECT DISTINCT exp.name FROM rollout r JOIN experiment exp USING(experiment_uid)
          WHERE r.state_uid=? AND r.eta_uid=? AND r.controller_uid=?
            AND r.conflict_quarantined=0 AND r.numerical_failure=0''',(st,eta,ctl)).fetchall()
        if not result:
            missing[role]+=1;continue
        cats={category(row[0]) for row in result}
        for cat in cats:count[(role,cat)]+=1
        for row in result:names[role][row[0]]+=1
    rows=[{'role':role,'category':cat,'pairs_with_provenance':n} for (role,cat),n in sorted(count.items())]
    with (OUT/'source_categories.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (OUT/'source_experiments.json').write_text(json.dumps({'by_role':{role:dict(cnt) for role,cnt in names.items()},
        'unmapped_exact_uid_pairs':dict(missing),'category_overlap_allowed':True},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pairs':len(pairs),'unmapped':dict(missing)}))

if __name__=='__main__':main()
