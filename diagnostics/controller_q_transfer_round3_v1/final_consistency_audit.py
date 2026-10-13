"""Read-only final counts, prereg hashes, and stage-status crosschecks."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(name):return json.loads((HERE/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipts={}
    for stage in ('P0','P1','P2PRIME','P3'):
        z=read(f'{stage}_REGISTRATION_RECEIPT.json')
        p=ROOT/z['path']
        assert p.is_file() and sha(p)==z['sha256'],stage
        receipts[stage]=dict(path=z['path'],utc=z['utc'],sha256=z['sha256'])
    p1=read('P1_POSTFLIGHT.json');p2=read('P2PRIME_POSTFLIGHT.json');r=read('results.json')
    assert p1['total']==21504 and p1['new_raw']==19968 and p1['missing']==p1['quarantined']==0
    assert p2['total']==67584 and p2['new_raw']==67584 and p2['valid']==66747 and p2['unknown']==837
    assert p2['missing']==p2['quarantined']==0 and p2['workers_done']
    assert r['total_new_rollouts']==87552 and not r['stages']['P2prime']['primary_confirmed']
    assert r['stages']['P3']['new_rollouts']==0
    assert (np.load(HERE/'P2PRIME_LABELS.npz')['outcomes']==-1).sum()==837
    assert len(list((HERE/'p1_raw').glob('*.json')))==19968
    assert len(list((HERE/'p2prime/raw').glob('*.json')))==67584
    assert all(not x['state_selection_gain_confirmed'] for x in read('P2PRIME_RESULTS.json')['qualifications'])
    assert read('P3_FEATURE_RECEIPT.json')['physical_steps']==0
    manifest=read('ARTIFACT_INDEX.json')
    for name in ('REPORT.md','HANDOFF.md','results.json'):
        assert manifest['keyfiles'][name]['sha256']==sha(HERE/name),name
    z=dict(status='PASS',preregistration_receipts=receipts,
           p1=dict(total=p1['total'],new=p1['new_raw'],valid=p1['valid'],unknown=p1['unknown']),
           p2prime=dict(total=p2['total'],new=p2['new_raw'],valid=p2['valid'],unknown=p2['unknown'],
                        missing=p2['missing'],quarantined=p2['quarantined']),
           new_rollouts=r['total_new_rollouts'],p2prime_primary_confirmed=False,
           p3_new_rollouts=0,artifact_index='ARTIFACT_INDEX.json')
    (HERE/'FINAL_CONSISTENCY_AUDIT.json').write_text(json.dumps(z,indent=2)+'\n')
    print(json.dumps(z))


if __name__=='__main__':main()
