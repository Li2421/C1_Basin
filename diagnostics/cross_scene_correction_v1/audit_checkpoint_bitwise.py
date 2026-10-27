"""Strict byte comparison of existing cross-process validation artifacts only."""
from common import *
import numpy as np


def main():
    original=read(HERE/'REPLAY_CHECK.json');assert original['passed']and original['bytewise_arrays']
    source=(HERE/'history.py').read_text()
    assert 'new.dtype == stored.dtype and new.shape == stored.shape and new.tobytes() == stored.tobytes()'in source
    pa=read(HERE/'CHECKPOINT_REPLAY_CHECK.json');cbs=read(HERE/'CHECKPOINT_PLANNER_SCOPE_CHECK.json')
    for proof in(pa,cbs):
        assert proof['passed']
        for name,expected in proof['implementation_sources'].items():assert sha(name)==expected,name
    rows=[]
    for label,samples in(('prioritized_astar',pa['rows']),('conflict_based_search',cbs['cbs_replay_rows'])):
        for sample in samples:
            folder=Path(sample['artifacts']);a_path=folder/'reference.npz';b_path=folder/'resumed.npz';fields=[]
            with np.load(a_path,allow_pickle=False)as a,np.load(b_path,allow_pickle=False)as b:
                assert set(a.files)==set(b.files)
                for name in a.files:
                    if name=='step_seconds':continue
                    left=a[name];right=b[name]
                    assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes(),('CHECKPOINT_STRICT_BITS_DIFFER',label,sample['scene'],name)
                    fields.append(name)
            rows.append(dict(controller=label,scene=sample['scene'],fields=fields,reference=str(a_path),
                reference_sha256=sha(a_path),resumed=str(b_path),resumed_sha256=sha(b_path),strict_bytes_equal=True))
    write(HERE/'CHECKPOINT_STRICT_BITWISE_AUDIT.json',dict(utc=utc(),passed=True,new_physical_steps=0,
        official_journal_writes=0,rows=rows,count=len(rows),excludes_only_wallclock_timing=True,
        strict_comparison='array dtype, shape, and raw C-order bytes; distinguishes signed zero.',
        original10_replays_already_used_raw_byte_comparison=True,
        original_replay_proof_sha256=sha(HERE/'REPLAY_CHECK.json'),original_comparison_source_sha256=sha(HERE/'history.py'),
        checkpoint_proof_sha256=sha(HERE/'CHECKPOINT_REPLAY_CHECK.json'),planner_scope_proof_sha256=sha(HERE/'CHECKPOINT_PLANNER_SCOPE_CHECK.json')))
    print('Strict checkpoint byte audit passed:',len(rows),'existing cross-process comparisons; no new physics.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(HERE/'CHECKPOINT_STRICT_BITS_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc()))
        raise
