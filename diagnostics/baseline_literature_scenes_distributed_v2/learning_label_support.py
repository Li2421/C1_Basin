"""Describe observed labels in the sealed menu; never tune or retrain a model."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import numpy as np
import collections


def main(seal_result=False):
    if (HERE/'LEARNING_LABEL_SUPPORT.receipt.json').exists():
        verify_receipt('LEARNING_LABEL_SUPPORT.receipt.json')
        return read(HERE/'LEARNING_LABEL_SUPPORT.json')
    rows=[];sources=[]
    for scene in NAMES:
        proof=f'LEARNING_BUNDLE_{scene}.receipt.json'
        if not (HERE/proof).exists():continue
        verify_receipt(proof);path=HERE/f'LEARNING_BUNDLE_{scene}.npz'
        with np.load(path) as z:labels=z['labels'];types=z['types']
        per_action=[dict(menu_index=i,success=int((labels[:,:,i]==1).sum()),
            failure=int((labels[:,:,i]==0).sum()),unknown=int((labels[:,:,i]<0).sum())) for i in range(12)]
        row=dict(scene=scene,shape=list(labels.shape),success=int((labels==1).sum()),
            failure=int((labels==0).sum()),unknown=int((labels<0).sum()),per_action=per_action,
            per_type={t:dict(success=int((labels[types==t]==1).sum()),failure=int((labels[types==t]==0).sum()),
                unknown=int((labels[types==t]<0).sum())) for t in sorted(set(types.tolist()))},
            observed_positive_action_support=bool((labels==1).any()))
        choice=HERE/f'LEARNED_EVAL_CHOICES_{scene}.json'
        if choice.exists():
            verify_receipt(f'LEARNED_EVAL_CHOICES_{scene}.receipt.json')
            values=read(choice)['choices_by_held_out_type']
            row['sealed_choice_counts']=dict(collections.Counter(str(a) for selected in values.values() for a in selected))
            sources.append(dict(path=str(choice),sha256=sha(choice)))
        sources.extend([dict(path=str(path),sha256=sha(path)),dict(path=str(HERE/proof),sha256=sha(HERE/proof))])
        rows.append(row)
    out=dict(utc=utc(),scope='Descriptive training-menu label support only; no independent-state or causal claim',
        scenes=rows,sources=sources,parameters_changed=False,new_rollouts=0)
    write(HERE/'LEARNING_LABEL_SUPPORT.json',out)
    if seal_result:
        seal('LEARNING_LABEL_SUPPORT.receipt.json',[HERE/'LEARNING_LABEL_SUPPORT.json',HERE/'learning_label_support.py'],
            descriptive_only=True,no_training_or_model_selection_changes=True)
    return out


if __name__=='__main__':main()
