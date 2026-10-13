"""Integrity checks on saved probes and frozen source provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cfg=json.loads((HARD/'screening_manifest.json').read_text())
    source={name:dict(expected=value,actual=sha(FIELD/name),matches=value==sha(FIELD/name))
            for name,value in cfg['controller_code_hashes'].items()}
    full_dirs=['results_full','boundary_results','boundary_normal_results','fixed_q_basin_results',
               'fixed_q_stable_basin_results','cross_boundary_results','cross_margin_gradient_results',
               'heldout_prediction_results']
    rows=[];max_phys=0.;key_bad=0;sign_bad=[];numerical=[]
    for d in ['results_h80']+full_dirs:
        folder=HERE/d
        for f in sorted(folder.glob('*.npz')):
            m=f.with_suffix('.json')
            if not m.exists():raise RuntimeError('missing metadata '+str(f))
            r=json.loads(m.read_text());z=np.load(f)
            pos=z['positions'];actions=z['actions'];keys=z['noise_keys']
            if len(pos)!=len(actions)+1:raise RuntimeError('trajectory length mismatch '+str(f))
            if len(actions)!=len(keys):raise RuntimeError('key length mismatch '+str(f))
            if len(actions):max_phys=max(max_phys,float(np.max(np.abs(np.diff(pos,axis=0)-.05*actions))))
            if r.get('termination')=='numerical_failure':numerical.append(str(m))
            if r.get('success') is not None and r.get('rho') is not None:
                if bool(r['rho']>0)!=bool(r['success']):sign_bad.append(str(m))
            rows.append(dict(path=str(f.relative_to(HERE)),sha256=sha(f),meta_sha256=sha(m),steps=len(actions)))
    result=dict(source_hashes=source,record_counts={d:sum(x['path'].startswith(d+'/') for x in rows) for d in ['results_h80']+full_dirs},
        total_saved_trials=len(rows),maximum_physical_update_residual=max_phys,
        numerical_failures=numerical,rho_success_sign_mismatches=sign_bad,
        has_raw_G_M=(HERE/'raw_G_M.npz').exists(),
        provenance=dict(frozen_confirmation_manifest=sha(HARD/'confirmation_manifest.json'),
            frozen_screening_manifest=sha(HARD/'screening_manifest.json'),
            original_local_operator_cache=sha(HARD/'transfer_mechanism_v2/state_dependence_control/manifest.json')),
        records=rows)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if not all(v['matches'] for v in source.values()) or sign_bad or numerical or max_phys>1e-9:raise RuntimeError('verification failed')


if __name__=='__main__':main()
