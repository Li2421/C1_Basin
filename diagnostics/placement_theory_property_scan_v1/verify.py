"""Audit raw probe counts, frozen sources, fixed keys, and physical updates."""
import hashlib
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    source=json.loads((HARD/'screening_manifest.json').read_text())['controller_code_hashes']
    code={name:dict(expected=h,actual=sha(FIELD/name),match=sha(FIELD/name)==h) for name,h in source.items()}
    pre={}
    for name in ('cache_preflight.json','live_full_cache_preflight.json','root_refine_cache_preflight.json'):
        x=json.loads((HERE/name).read_text())['summary'];pre[name]=x
        if x['ambiguous']:raise RuntimeError('Ambiguous preflight '+name)
    counts={};hashes={};mismatch=[];bad_update=[];max_resid=0.;errors=[]
    for d,n in (('action_results',72),('state_results',144),('rollout_results',280)):
        fs=sorted((HERE/d).glob('*.json'));counts[d]=len(fs)
        if len(fs)!=n:raise RuntimeError(f'{d}: {len(fs)} instead of {n}')
        h=hashlib.sha256()
        for f in fs:
            z=f.with_suffix('.npz')
            if not z.exists():raise RuntimeError('missing npz '+str(z))
            h.update(f.name.encode());h.update(bytes.fromhex(sha(f)))
            h.update(z.name.encode());h.update(bytes.fromhex(sha(z)))
            r=json.loads(f.read_text())
            if 'error' in r and d!='rollout_results':errors.append(str(f))
            if d!='rollout_results':continue
            with np.load(z) as a:
                p=a['positions'];u=a['actions'];v=a['velocities'];keys=a['noise_keys']
                if len(p)!=len(u)+1 or len(p)!=len(v) or len(keys)!=len(u):
                    raise RuntimeError('trace shape '+str(f))
                if len(u):
                    resid=float(np.max(np.abs(np.diff(p,axis=0)-.05*u)))
                    max_resid=max(max_resid,resid)
                    if resid>1e-9:bad_update.append(str(f))
            if r['success'] is not None and r['termination']!='numerical_failure' and r['rho'] is not None:
                if bool(r['rho']>0)!=bool(r['success']):mismatch.append(str(f))
        hashes[d]=h.hexdigest()
    pairs=[]
    rp=json.loads((HERE/'rollout_protocol.json').read_text())
    for j in rp['jobs']:
        if j['kind']=='basis_feedback' and j['mode']=='frozen_B':
            base=f"{j['uid']}_{j['chain']}_c{j['candidate_index']}_"
            a=HERE/'rollout_results'/(base+'frozen_B.npz')
            b=HERE/'rollout_results'/(base+'live_B_full.npz')
            with np.load(a) as x,np.load(b) as y:
                n=min(len(x['noise_keys']),len(y['noise_keys']))
                pairs.append(dict(pair=base,shared_steps=n,
                    same=bool(np.array_equal(x['noise_keys'][:n],y['noise_keys'][:n]))))
    result=dict(source_hashes=code,preflight=pre,record_counts=counts,directory_sha256=hashes,
        action_probe_errors=errors,physical_update_bad=bad_update,max_physical_update_residual=max_resid,
        rho_success_sign_mismatch=mismatch,live_frozen_key_pairs=pairs,
        pilot_exclusion=json.loads((HERE/'pilot_invalid.json').read_text()))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(counts=counts,max_physical_update_residual=max_resid,
        action_probe_errors=len(errors),physical_update_bad=len(bad_update),
        rho_success_sign_mismatch=len(mismatch),key_pair_mismatch=sum(not x['same'] for x in pairs)),indent=2))
    if not all(x['match'] for x in code.values()) or errors or bad_update or mismatch or any(not x['same'] for x in pairs):
        raise RuntimeError('verification failed')

if __name__=='__main__':main()
