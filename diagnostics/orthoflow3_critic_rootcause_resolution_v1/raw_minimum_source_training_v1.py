"""One joint ablation, reusing the identical source training implementation."""
import argparse
from pathlib import Path
from . import raw_input_ablation as inherited
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'raw_minimum_source'
RULE=ROOT/'raw_minimum_protocol.json'


def configure():
    inherited.OUT,inherited.RULE=OUT,RULE
    inherited.ARMS=('without_explicit_h',)
    original=inherited.data.load
    def load(fit):
        d,x,tr,va,norm=original(fit)
        d={**d,'input_context':d['input_context'].copy()}
        d['input_context'][:,:,:88]=0
        return d,x,tr,va,norm
    inherited.data.load=load


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0)
    a=p.parse_args();configure()
    if a.action=='train':
        inherited.train(a.index)
        path=OUT/'without_explicit_h'/f'fold{a.index//3}/seed{inherited.base.SEEDS[a.index%3]}'/'joint_deletion_wrapper.json'
        inherited.write(path,dict(wrapper_sha256=inherited.sha(__file__),protocol_sha256=inherited.sha(RULE),
            underlying_code_sha256=inherited.sha(inherited.__file__),input_context_channels_zeroed=[0,88],
            explicit_h_zeroed=True,entity_masks_preserved=True,target_labels_used=False,new_rollouts=0))
    else:
        for f in range(3):
            for s in inherited.base.SEEDS:
                d=inherited.read(OUT/'without_explicit_h'/f'fold{f}/seed{s}/joint_deletion_wrapper.json')
                assert d['wrapper_sha256']==inherited.sha(__file__) and d['underlying_code_sha256']==inherited.sha(inherited.__file__)
        inherited.summarize()
