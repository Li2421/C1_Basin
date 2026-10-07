"""Reuses identical all-source final fitting; only initial response is masked."""
import argparse
from pathlib import Path
from . import raw_bypass_final as fit
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'minimal_response_final_models'
RULE=ROOT/'minimal_response_final_protocol.json'


def configure():
    fit.OUT,fit.RULE=OUT,RULE
    original=fit.data.load
    def load(which):
        d,x,tr,va,norm=original(which)
        d={**d,'input_context':d['input_context'].copy()}
        d['input_context'][:,:,:88]=0
        return d,x,tr,va,norm
    fit.data.load=load
    # Both independently source-selected stop times are200; no target tuning.
    selection=fit.read(ROOT/'raw_input_ablation/selection_frozen.json')['selected']['without_initial_response']
    assert selection['step']==fit.read(fit.skip.OUT/'selection_frozen.json')['selected']['step']==200


def train(index):
    assert index in (0,1,2)
    fit.train(index+3)
    path=OUT/'raw_bypass'/f'seed{fit.base.SEEDS[index]}'/'input_mask_audit.json'
    fit.write(path,dict(wrapper_sha256=fit.sha(__file__),underlying_fitter_sha256=fit.sha(fit.__file__),
        source_selection_sha256=fit.sha(ROOT/'raw_input_ablation/selection_frozen.json'),
        context_zero_channels=[0,88],explicit_h_retained=True,goal_response_channels=32,
        target_labels_used=False,new_rollouts=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();configure();train(a.index)
