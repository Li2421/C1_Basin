#!/usr/bin/env python3
"""Select conservative penalty exclusively on frozen 12-state VAL panel."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
import common
HERE=Path(__file__).resolve().parent
SRC=common.ROOT/'diagnostics/orthoflow3_structured_continuous_q_data_v1/structured_pair_table.parquet'
LAMBDA=[0.,.25,.5,1.,2.]

def main():
    rows=[r for r in pq.read_table(SRC).to_pylist() if r['matrix_partition']=='VAL_VAL']
    rows.sort(key=lambda r:(r['state_uid'],r['eta_uid']))
    assert len(rows)==144 and len({r['state_uid'] for r in rows})==12
    h=np.asarray([r['h_raw'] for r in rows],np.float32)
    eta=np.asarray([r['eta'] for r in rows],np.float32)
    q=np.asarray([r['empirical_q'] for r in rows],np.float64)
    pred=common.predict_members(h,eta)
    mu=pred.mean(0);sigma=pred.std(0)
    by=defaultdict(list)
    for i,r in enumerate(rows):by[r['state_uid']].append(i)
    groups=list(by.values());assert all(len(ix)==12 for ix in groups)
    candidates=[]
    for lam in LAMBDA:
        chosen=[max(ix,key=lambda j:(mu[j]-lam*sigma[j],-j)) for ix in groups]
        oracle=[max(q[ix]) for ix in groups]
        selected=[q[j] for j in chosen]
        candidates.append({'lambda':lam,'mean_selected_empirical_Q':float(np.mean(selected)),
          'mean_regret':float(np.mean(np.asarray(oracle)-np.asarray(selected))),
          'strong_success_proxy_Q_ge_0p9375':int(sum(v>=15/16 for v in selected)),
          'selected_indices':chosen})
    winner=sorted(candidates,key=lambda r:(-r['mean_selected_empirical_Q'],r['mean_regret'],r['lambda']))[0]
    (HERE/'val_lambda_grid.json').write_text(json.dumps({'grid':candidates,'n_val_states':12,
      'VAL_labels_mostly_8_seed_proxy_not_B15_certification':True},indent=2)+'\n')
    (HERE/'frozen_selection.json').write_text(json.dumps({'lambda':winner['lambda'],
      'criterion':'VAL mean empirical selected Q, regret, smaller lambda',
      'selected_before_hard_test_evaluation':True,
      'val_selected_Q':winner['mean_selected_empirical_Q']},indent=2)+'\n')
    print(json.dumps({'selected_lambda':winner['lambda'],'val_Q':winner['mean_selected_empirical_Q'],
      'grid':[{k:v for k,v in r.items() if k!='selected_indices'} for r in candidates]}))
if __name__=='__main__':main()
