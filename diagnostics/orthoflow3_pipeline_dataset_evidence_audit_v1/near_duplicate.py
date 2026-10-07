#!/usr/bin/env python3
"""Feature-neighbor audit; detects near-duplicates without outcome access."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT=Path('/home/zhihan/research/Basin_C1')
OUT=Path(__file__).resolve().parent
BASE=ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1'
STRUCT=ROOT/'diagnostics/orthoflow3_structured_continuous_q_data_v1'
rows=pq.read_table(BASE/'pair_table.parquet').to_pylist()
manifest=json.loads((BASE/'dataset_manifest.json').read_text())
result={}
for scen in ('Toy','DB'):
    arr={}
    for sp in ('train','val','test'):
        z={r['state_uid']:r['h_raw'] for r in rows if r['scenario']==scen and r['state_split']==sp}
        arr[sp]=np.asarray(list(z.values()),float)
    m=np.asarray(manifest['state_normalization'][scen]['mean'])
    s=np.asarray(manifest['state_normalization'][scen]['std'])
    for a,b in (('train','test'),('train','val'),('val','test')):
        dd=cKDTree((arr[a]-m)/s).query((arr[b]-m)/s)[0]/np.sqrt(arr[a].shape[1])
        result[f'generator_{scen}_{a}_vs_{b}']={'min_rms':float(dd.min()),'median_rms':float(np.median(dd)),
            'p05_rms':float(np.percentile(dd,5)),'n_below_0p05':int(np.sum(dd<.05)),
            'n_below_0p10':int(np.sum(dd<.1)),'query_states':len(dd)}
cr=pq.read_table(STRUCT/'structured_pair_table.parquet').to_pylist()
a={}
for sp in ('TRAIN_TRAIN','VAL_VAL','TESTSTATE_TESTETA'):
    z={r['state_uid']:r['h_raw'] for r in cr if r['matrix_partition']==sp}
    a[sp]=np.asarray(list(z.values()),float)
m=a['TRAIN_TRAIN'].mean(axis=0);s=np.maximum(a['TRAIN_TRAIN'].std(axis=0),1e-6)
dd=cKDTree((a['TRAIN_TRAIN']-m)/s).query((a['TESTSTATE_TESTETA']-m)/s)[0]/np.sqrt(a['TRAIN_TRAIN'].shape[1])
result['critic_Toy_train_vs_test']={'min_rms':float(dd.min()),'median_rms':float(np.median(dd)),
    'p05_rms':float(np.percentile(dd,5)),'n_below_0p05':int(np.sum(dd<.05)),
    'n_below_0p10':int(np.sum(dd<.1)),'query_states':len(dd)}
(OUT/'near_duplicate_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result))
