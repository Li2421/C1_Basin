"""A locked controller choice depends only on frozen parameters and probe features."""
from common import *
from selector import Regression,ARMS
from features import NAMES
from selector_candidates_v2 import restore


def main():
    assert not (HERE/'LOCKED_CHOICES.receipt.json').exists()
    receipt=read(HERE/'FROZEN_PIPELINE.receipt.json')
    for path,digest in receipt['files'].items():assert sha(path)==digest
    frozen=read(HERE/'FROZEN_PIPELINE.json')
    with np.load(HERE/'LOCKED_FEATURES.npz',allow_pickle=False) as z:
        assert not set(z.files)&{'labels','success','tuning','outcome'}
        assert list(z['feature_names'])==list(NAMES)
        assert set(z['types']).issubset({'ring_ippo','implicit'})
        names=z['names'];types=z['types'];x=z['features'];r5=z['r5']
    model=restore(frozen['model']);comparison=restore(frozen['r5_comparator'])
    if not (HERE/'LOCKED_COMPARATOR_CHOICES.receipt.json').exists():
        from seal_comparators import main as seal_blind_comparators
        seal_blind_comparators('locked')
    choice=model.choose(x);r5choice=comparison.choose(r5)
    rows=[dict(controller=str(n),type=str(t),arm=ARMS[int(a)],r5_arm=ARMS[int(b)],
               unavailable_probe_fallback=bool(not np.isfinite(features).any()))
          for n,t,a,b,features in zip(names,types,choice,r5choice,x)]
    write(HERE/'LOCKED_CHOICES.json',dict(utc=utc(),controllers=rows,locked_eval_or_tuning_labels_read=False,
           frozen_pipeline_sha256=sha(HERE/'FROZEN_PIPELINE.json')))
    seal('LOCKED_CHOICES.receipt.json',[HERE/'LOCKED_CHOICES.json',HERE/'LOCKED_FEATURES.npz',
          HERE/'FROZEN_PIPELINE.json',HERE/'choose_locked.py',HERE/'LOCKED_COMPARATOR_CHOICES.receipt.json'],before_first_locked_eval_or_tuning_submission=True)
    print('Locked choices sealed:',len(rows),'controllers; no eval/tuning labels read')

if __name__=='__main__':main()
