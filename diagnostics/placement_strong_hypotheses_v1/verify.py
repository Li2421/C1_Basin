"""Integrity audit of frozen source, preflighted requests, full traces and labels."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
RESULT_DIRS=dict(panel_results=2536,grid9_results=2416,robust_results=1920,
                 sobol128_results=3072,
                 curvature_results=128,curvature_followup_results=128,hybrid_results=36,
                 hybrid_grid_results=170,h1_line_results=180,h1_refine_results=36,
                 h1_gradient_results=104)
PREFLIGHTS=('cache_preflight.json','grid9_cache_preflight.json','robust_cache_preflight.json',
            'sobol128_cache_preflight.json',
            'curvature_cache_preflight.json','curvature_followup_cache_preflight.json',
            'hybrid_cache_preflight.json','hybrid_grid_cache_preflight.json',
            'h1_line_cache_preflight.json','h1_gradient_cache_preflight.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=json.loads((HARD/'screening_manifest.json').read_text())['controller_code_hashes']
    source_check={name:dict(expected=h,actual=sha(FIELD/name),match=sha(FIELD/name)==h) for name,h in source.items()}
    preflight={}
    for name in PREFLIGHTS:
        r=json.loads((HERE/name).read_text());preflight[name]=r['summary']
        if r['summary']['ambiguous']:raise RuntimeError('ambiguous cache request '+name)
    for i in range(1,4):
        name=f'h1_refine_round{i}_cache_preflight.json';r=json.loads((HERE/name).read_text());preflight[name]=r['summary']
        if r['summary']['ambiguous']:raise RuntimeError('ambiguous refine request')
    counts={};directory_hashes={};terms=Counter();numeric=[];sign_bad=[];physical_bad=[];max_resid=0.
    for d,expected in RESULT_DIRS.items():
        fs=sorted((HERE/d).glob('*.json'));counts[d]=len(fs)
        digest=hashlib.sha256()
        if len(fs)!=expected:raise RuntimeError(f'{d} expected {expected}, got {len(fs)}')
        for f in fs:
            r=json.loads(f.read_text());npz=f.with_suffix('.npz')
            if not npz.exists():raise RuntimeError('missing trace '+str(npz))
            digest.update(f.name.encode());digest.update(bytes.fromhex(sha(f)))
            digest.update(npz.name.encode());digest.update(bytes.fromhex(sha(npz)))
            with np.load(npz) as z:
                p=z['positions'];u=z['actions'];v=z['velocities'];keys=z['noise_keys']
                if len(p)!=len(u)+1 or len(v)!=len(p) or len(keys)!=len(u):raise RuntimeError('trace length '+str(f))
                if len(u):
                    resid=float(np.max(np.abs(np.diff(p,axis=0)-.05*u)))
                    max_resid=max(max_resid,resid)
                    if resid>1e-9:physical_bad.append(str(f))
            term=r['termination'];terms[(d,term)]+=1
            if term=='numerical_failure':numeric.append(str(f))
            elif r.get('rho') is not None and r.get('success') is not None and (r['rho']>0)!=r['success']:
                sign_bad.append(str(f))
        directory_hashes[d]=digest.hexdigest()
    def check_pairs(d,pairs):
        total=0;bad=[]
        for a,b in pairs:
            with (np.load(HERE/d/a) as x,np.load(HERE/d/b) as y):
                xx=x['noise_keys'];yy=y['noise_keys'];n=min(len(xx),len(yy))
                if not np.array_equal(xx[:n],yy[:n]):bad.append((a,b))
            total+=1
        return dict(checked=total,mismatches=bad)
    panel=json.loads((HERE/'protocol.json').read_text())
    pp=[]
    for st in panel['states']:
        uid=st['uid'];kinds=['sobol64']+(['grid5'] if uid in panel['grid_states'] else [])
        for kind in kinds:
            n=64 if kind=='sobol64' else 125
            pp += [(f'{uid}_{kind}_{i:03d}_TT.npz',f'{uid}_{kind}_{i:03d}_FF.npz') for i in range(n)]
    paired=dict(panel=check_pairs('panel_results',pp))
    grid9=json.loads((HERE/'grid9_protocol.json').read_text())
    gg=[]
    for st in grid9['states']:
        uid=st['uid']
        for ix in {(tuple(j['index'])) for j in grid9['jobs'] if j['uid']==uid}:
            key=''.join(map(str,ix));gg.append((f'{uid}_TT_{key}.npz',f'{uid}_FF_{key}.npz'))
    paired['grid9']=check_pairs('grid9_results',gg)
    qmc=json.loads((HERE/'sobol128_protocol.json').read_text())
    qq=[(f"{st['uid']}_sobol128b_{i:03d}_TT.npz",f"{st['uid']}_sobol128b_{i:03d}_FF.npz")
        for st in qmc['states'] for i in range(128)]
    paired['sobol128']=check_pairs('sobol128_results',qq)
    rr=[]
    for st in json.loads((HERE/'robust_protocol.json').read_text())['states']:
        uid=st['uid']
        rr += [(f'{uid}_TT_s{i:02d}_seed{s:02d}.npz',f'{uid}_FF_s{i:02d}_seed{s:02d}.npz')
               for i in range(32) for s in range(1,16)]
    paired['robust']=check_pairs('robust_results',rr)
    lines=json.loads((HERE/'h1_line_protocol.json').read_text())['lines']
    ll=[(f"{line['uid']}_TT_t{i:02d}.npz",f"{line['uid']}_FF_t{i:02d}.npz")
        for line in lines if line['available'] for i in range(1,16)]
    paired['h1_lines']=check_pairs('h1_line_results',ll)
    for name,proto,d in (('hybrid','hybrid_protocol.json','hybrid_results'),
                         ('hybrid_grid','hybrid_grid_protocol.json','hybrid_grid_results')):
        p=json.loads((HERE/proto).read_text())
        ids={(j['uid'],j['index']) for j in p['jobs']}
        hh=[(f'{uid}_FF_base_TT_corr_{i:02d}.npz',f'{uid}_TT_base_FF_corr_{i:02d}.npz') for uid,i in ids]
        paired[name]=check_pairs(d,hh)
    result=dict(source_hashes=source_check,preflight=preflight,record_counts=counts,
                directory_sha256=directory_hashes,
                total_new_traces=sum(counts.values()),termination_counts={str(k):v for k,v in terms.items()},
                numerical_failure_count=len(numeric),numerical_failure_paths=numeric,
                valid_rho_sign_mismatch_count=len(sign_bad),valid_rho_sign_mismatch_paths=sign_bad,
                maximum_physical_update_residual=max_resid,physical_update_failures=physical_bad,
                paired_flow_keys=paired)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('numerical_failure_paths','termination_counts','preflight')},indent=2))
    if not all(x['match'] for x in source_check.values()) or sign_bad or physical_bad or any(x['mismatches'] for x in paired.values()):
        raise RuntimeError('verification failed')


if __name__=='__main__':main()
