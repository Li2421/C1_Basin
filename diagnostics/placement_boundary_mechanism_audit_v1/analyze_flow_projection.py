"""Read pre-existing fixed-key ambient M_flow / M_FF decomposition anchors."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'field_transfer_operator_audit_v3/fixed_noise_operators.json'

def main():
    data=json.loads(SOURCE.read_text());out={}
    for scene in ('toy_give_way','ring_exchange'):
        rows=[r for r in data if r['scene']==scene]
        mf=np.asarray([r['M_flow'] for r in rows]);mp=np.asarray([r['M_FF'] for r in rows])
        effect=np.linalg.norm(mp-mf,axis=(1,2))/np.maximum(np.linalg.norm(mp,axis=(1,2)),1e-12)
        center=mf.mean(0);disp=np.sqrt(np.mean((np.linalg.norm(mf-center,axis=(1,2))/np.linalg.norm(center))**2))
        out[scene]=dict(n=len(rows),state_uids=[r['state_uid'] for r in rows],
            flow_state_dispersion=float(disp),field_projection_effect_median=float(np.median(effect)),
            field_projection_effect_max=float(np.max(effect)),effect_by_state=effect.tolist())
    (HERE/'flow_projection_decomposition.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:{j:v[j] for j in ('n','flow_state_dispersion','field_projection_effect_median')} for k,v in out.items()},indent=2))
if __name__=='__main__':main()
