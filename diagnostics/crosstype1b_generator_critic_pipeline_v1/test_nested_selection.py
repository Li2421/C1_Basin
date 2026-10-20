"""Outer-label perturbation must not affect source-inner checkpoint choice."""
from core import *
import select_models as selection


class FakePath:
    def __init__(self,source,leaf=''):self.source=tuple(source);self.leaf=leaf
    def __truediv__(self,leaf):return FakePath(self.source,leaf)
    def exists(self):return True
    def __str__(self):return '/synthetic/'+','.join(self.source)+'/'+self.leaf


def main():
    source=['a','b','c'];outer='held_outer';perturb=[0.]
    original_path,original_read=selection.path_for,selection.read
    def read_fake(path):
        if path.leaf=='TASK.json':return dict(source_types=list(path.source))
        omitted=set(source)-set(path.source);assert len(omitted)==1
        held=next(iter(omitted))
        return [dict(step=step,validation={held:dict(nll=(.4 if step==100 else .7),ranking_accuracy=.6),
            outer:dict(nll=perturb[0]+(1000 if step==100 else -1000),ranking_accuracy=perturb[0])}) for step in (100,200)]
    try:
        selection.path_for=lambda fit_source,mode,init:FakePath(fit_source)
        selection.read=read_fake
        before=selection.inner_selection(source,'full',dict(steps=[100,200]))
        perturb[0]=1e6
        after=selection.inner_selection(source,'full',dict(steps=[100,200]))
        assert before==after
        assert before['selected']['step']==100 and before['selected']['initialization']=='scratch'
        assert set(before['selected']['source_only_folds'])==set(source)
    finally:selection.path_for,selection.read=original_path,original_read
    write(HERE/'NESTED_SELECTION_TESTS.json',dict(utc=utc(),passed=True,
        source_only_inner_labels=True,outer_validation_perturbation_exactly_invariant=True,tie_prefers_scratch=True))
    seal('NESTED_SELECTION_TESTS.receipt.json',[HERE/'NESTED_SELECTION_TESTS.json',HERE/'test_nested_selection.py',HERE/'select_models.py'])
    print('Nested source-only selection checks passed',flush=True)


if __name__=='__main__':main()
