"""Finite-amplitude error at cached common trajectory states, with identical action noise."""
import collections
import numpy as np
from cache_analysis import CACHE,read,save,stats,sha,CHAINS

def main():
    tasks=[t for t in read(CACHE/'tasks.json')['tasks'] if t['kind']!='initial']
    journal={}
    for p in (CACHE/'journals').glob('*.jsonl'):
        import json
        for line in p.read_text().splitlines():
            r=json.loads(line);journal[r['id']]=r
    rows=[]
    for t in tasks:
        path=CACHE/'results'/t['id']
        assert sha(path.with_suffix('.npz'))==journal[t['id']]['npz_sha256']
        data=read(path.with_suffix('.json'));z=np.load(path.with_suffix('.npz'))
        for r in data['rows']:
            prefix=r['array_prefix'];cx=prefix.split('_center')[0];eta=np.asarray(r['eta'])
            out=z[cx+'_exec_all'];jac=z[prefix+'_J_eta_exec'][:,2]
            for ci,ch in enumerate(CHAINS):
                q=r['chains'][ch]
                reliable=bool(q['fd_stable'] and q['chain_rule_relative_error'] is not None and q['chain_rule_relative_error']<=.05)
                actual=out[ci,1]-out[ci,0];prediction=jac[ci]@eta;e=np.linalg.norm(actual-prediction)
                rows.append(dict(scene=r['scene'],phase=r['phase'],state_uid=r['state_uid'],seed=r['seed'],eta_index=r['eta_index'],chain=ch,
                                 reliable=reliable,error=e,response_norm=np.linalg.norm(actual),relative_error=e/max(np.linalg.norm(actual),1e-8)))
        z.close()
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['scene']+':'+r['phase']+':'+r['chain']].append(r)
    summary={}
    for key,allrows in grouped.items():
        good=[r for r in allrows if r['reliable']]
        summary[key]=dict(total=len(allrows),reliable=len(good),error=stats([r['error'] for r in good]),
                          relative_error=stats([r['relative_error'] for r in good]),
                          over25pct=np.mean([r['relative_error']>.25 for r in good]))
    save('trajectory_linearization.json',dict(npz_hashes_checked=len(tasks),summary=summary))

if __name__=='__main__':main()
