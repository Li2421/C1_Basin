#!/usr/bin/env python3
"""Read-only audit of existing DB hard-panel off-anchor proposal evidence."""
import csv,json,statistics
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=Path(__file__).resolve().parent
PRIOR=ROOT/'diagnostics/orthoflow3_db_mode_free_mustdo_v1'
frozen=json.loads((PRIOR/'db_frozen_proposals.json').read_text())['states']
q={(r['state_id'],int(r['sample_index'])):r for r in csv.DictReader((PRIOR/'db_proposal_q16.csv').open())}
rows=[]
for s in frozen:
    sid=s['state_id'];eta=np.asarray(s['eta'],float)
    robust=np.array([int(q[(sid,j)]['B15']) for j in range(16)],bool)
    Q=np.array([float(q[(sid,j)]['Q16']) for j in range(16)])
    # Normalize by the frozen generator's full per-axis E_bridge widths.
    scale=2*np.asarray([0.625,0.5,0.375],float)
    pos=eta[robust]/scale;neg=eta[~robust]/scale
    dist=np.linalg.norm(pos[:,None,:]-neg[None,:,:],axis=2) if len(pos) and len(neg) else None
    j=s['critic_choice_K16']
    rows.append({'state_id':sid,'n_proposals':16,'B15_proposals':int(robust.sum()),
                 'mean_Q16':float(Q.mean()),'min_Q16':float(Q.min()),'max_Q16':float(Q.max()),
                 'success_failure_coexist':int(len(pos)>0 and len(neg)>0),
                 'nearest_robust_to_failure_norm':float(dist.min()) if dist is not None else '',
                 'boundary_pair_within_0p10':int(dist is not None and dist.min()<=.10),
                 'critic_choice_B15':int(robust[j]),'critic_choice_Q16':float(Q[j])})
with (HERE/'db_offanchor_boundary_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'states':len(rows),'proposal_points':len(rows)*16,
         'B15_proposal_fraction':sum(r['B15_proposals'] for r in rows)/(len(rows)*16),
         'B15_proposals_per_state_median':statistics.median(r['B15_proposals'] for r in rows),
         'states_with_B15_and_failure_proposals':sum(r['success_failure_coexist'] for r in rows),
         'states_with_success_failure_within_norm_0p10':sum(r['boundary_pair_within_0p10'] for r in rows),
         'states_with_<=4_B15_proposals':sum(r['B15_proposals']<=4 for r in rows),
         'critic_misselection_states':[r['state_id'] for r in rows if not r['critic_choice_B15']]}
(HERE/'db_offanchor_boundary_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
