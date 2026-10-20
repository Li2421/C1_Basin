"""User-authorized m=1 source-controller inner validation; outer labels excluded."""
from common import *
from train_critic import key_for
from critic_model import metrics


def controller_key(source,name):
    return hashlib.sha256(json.dumps(['m1-controller-held',sorted(source),name]).encode()).hexdigest()[:20]


def folder(source,name,mode,init):
    return HERE/'critic_fits'/f'{mode}_{init}'/controller_key(source,name)


def choose(source,mode,configuration,data,initialization=None):
    assert len(source)==1
    names=data['names'][data['types']==source[0]];rows=[];histories={}
    for init in ((initialization,) if initialization else ('scratch','warm')):
        for name in names:
            path=folder(source,str(name),mode,init);assert (path/'COMPLETE.receipt.json').exists()
            task=read(path/'TASK.json');assert task['held_controller']==name and task['source_types']==source
            histories[(init,str(name))]={r['step']:r['validation']['__held_controller__'+str(name)] for r in read(path/'HISTORY.json')}
        for step in configuration['steps']:
            folds={str(n):histories[(init,str(n))][step] for n in names};ranks=[v['ranking_accuracy'] for v in folds.values() if v['ranking_accuracy'] is not None]
            rows.append(dict(initialization=init,step=step,mean_nll=float(np.mean([v['nll'] for v in folds.values()])),mean_ranking_accuracy=float(np.mean(ranks)) if ranks else None,source_controller_folds=folds))
    pick=min(rows,key=lambda r:(round(r['mean_nll'],12),r['step'],-(r['mean_ranking_accuracy'] or 0),r['initialization']!='scratch'))
    return dict(selected=pick,history=rows,source_types=source,source_controller_inner=True,outer_labels_used=False)


def held_predictions(source,mode,selection,data):
    chosen=selection['selected'];probability=None;types=data['types'].astype('U160').copy()
    for name in data['names'][data['types']==source[0]]:
        path=folder(source,str(name),mode,chosen['initialization']);packet=path/f"step{chosen['step']}_predictions.npz";receipt=read(path/'COMPLETE.receipt.json')
        assert sha(packet)==receipt['files'][str(packet)]
        with np.load(packet,allow_pickle=False) as z:p=z['probability'];assert np.array_equal(z['names'],data['names'])
        if probability is None:probability=np.zeros_like(p)
        mask=data['names']==name;probability[:,mask]=p[:,mask]
        # Only each held source controller's predictions will enter NLL.
    return probability
