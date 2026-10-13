"""Quantify placement-dependent zero-correction offsets as a competing path."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
CACHE=Path('/home/zhihan/research/Basin_C1_hard_field_20261006/transfer_mechanism_v2/state_dependence_control')

def main():
    manifest=json.loads((CACHE/'manifest.json').read_text());rows=[]
    for uid in manifest['states']:
        z=np.load(CACHE/f'{uid}.npz');tt=z['exec_all'][0,0];ff=z['exec_all'][3,0]
        rows.append(dict(uid=uid,scene='toy_give_way' if 'toy' in uid else 'ring_exchange',
            initial_action_offset=float(np.linalg.norm(ff-tt)),
            initial_action_offset_relative_to_TT=float(np.linalg.norm(ff-tt)/max(np.linalg.norm(tt),1e-12))))
    for uid in ['hard_confirm_toy_1_820001','hard_confirm_toy_1_820013','hard_confirm_ring_1_920001','hard_confirm_ring_1_920013']:
        a=np.load(HERE/'results_h80'/f'{uid}_TT_zero_center.npz');b=np.load(HERE/'results_h80'/f'{uid}_FF_zero_center.npz')
        row=next(x for x in rows if x['uid']==uid)
        row['position_baseline_separation_t80']=float(np.linalg.norm(a['positions'][80]-b['positions'][80]))
    (HERE/'baseline_offsets.json').write_text(json.dumps(rows,indent=2)+'\n')
    for scene in ('toy_give_way','ring_exchange'):
        vals=[x['initial_action_offset'] for x in rows if x['scene']==scene]
        print(scene,len(vals),float(np.median(vals)),float(np.max(vals)))
if __name__=='__main__':main()
