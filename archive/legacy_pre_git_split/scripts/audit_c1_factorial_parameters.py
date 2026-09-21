"""Check saved line-search parameters against the declared update, no rollout."""
import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def leaves(tree,prefix=()):
    if isinstance(tree,Mapping):
        result={}
        for key in sorted(tree):result.update(leaves(tree[key],prefix+(key,)))
        return result
    return {prefix:np.asarray(tree)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder
    protocol=json.loads((folder/'protocol.json').read_text())
    diagnostics=json.loads((folder/'diagnostics.json').read_text())
    index={(d['rid'],d['risk']):d for d in diagnostics}
    assert len(index)==len(diagnostics)
    checked=[];reference_origin=None;max_error=0.
    for path in sorted(folder.glob('local_params_*.pkl')):
        saved=pickle.loads(path.read_bytes());rid=saved['rid'];policies=saved['params']
        np.testing.assert_array_equal(saved['initial'],protocol['starts'][rid])
        reference=leaves(policies['reference'])
        assert all(np.isfinite(x).all() for x in reference.values())
        if reference_origin is None:reference_origin=reference
        for key in reference:np.testing.assert_array_equal(reference[key],reference_origin[key])
        for name in ('ordered','guarded'):
            d=index[rid,name];alpha=d['initial_alpha'];accepted=d['accepted_alpha']
            assert 0<=accepted<=alpha
            if accepted:
                assert d['trials'][-1]['alpha']==accepted
                assert d['trials'][-1]['risk']<d['before']-1e-9*max(1.,abs(d['before']))
            fixed=leaves(policies[name+'_fixed']);backtracked=leaves(policies[name+'_backtrack'])
            assert set(fixed)==set(reference)==set(backtracked)
            for key,base in reference.items():
                expected=base+(accepted/alpha)*(fixed[key]-base) if alpha else base
                np.testing.assert_allclose(backtracked[key],expected,atol=1e-12,rtol=1e-12)
                max_error=max(max_error,float(np.max(np.abs(backtracked[key]-expected))))
                if not accepted:np.testing.assert_array_equal(backtracked[key],base)
        checked.append(dict(rid=rid,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(scope='parameter transaction audit only; not gradient correctness or efficacy',
        checked_starts=len(checked),planned_starts=len(protocol['starts']),
        complete_scope=len(checked)==len(protocol['starts']),
        max_parameter_interpolation_error=max_error,checked=checked,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'parameter_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checked'},indent=2))


if __name__=='__main__':main()
